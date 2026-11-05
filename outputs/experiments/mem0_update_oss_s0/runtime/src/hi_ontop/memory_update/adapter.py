"""Replace one call in Mem0 0.1.118; leave extraction, search and storage intact."""
from __future__ import annotations

import ast
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import re
import textwrap
import time

os.environ['MEM0_TELEMETRY'] = 'false'

from mem0 import Memory
import mem0.memory.main as upstream
from mem0.configs.base import MemoryConfig
from mem0.configs.prompts import FACT_RETRIEVAL_PROMPT
from mem0.memory.storage import SQLiteManager
from mem0.vector_stores.qdrant import Qdrant
from qdrant_client import QdrantClient
import torch

from .common import turn_texts
from .model import OPS, labels

FIXED_EXTRACTION_PROMPT = re.sub(r"Today's date is \d{4}-\d{2}-\d{2}",
                                 "Today's date is 2026-09-26", FACT_RETRIEVAL_PROMPT)
_memory_context = ContextVar('mem0_update_instance', default=None)


def patched_add():
    if importlib.metadata.version('mem0ai') != '0.1.118':
        raise RuntimeError('This hook requires mem0ai==0.1.118')
    source = textwrap.dedent(inspect.getsource(Memory._add_to_vector_store))
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'response':
            if not isinstance(node.value, ast.Call) or 'function_calling_prompt' not in ast.unparse(node.value):
                continue
            node.value = ast.parse('self.decide_update(new_retrieved_facts, retrieved_old_memory, '
                                   'function_calling_prompt, temp_uuid_mapping)').body[0].value
            count += 1
    if count != 1:
        raise RuntimeError(f'Expected exactly one update call, found {count}')
    namespace = dict(vars(upstream))
    exec(compile(ast.fix_missing_locations(tree), '<mem0-update-hook>', 'exec'), namespace)
    return namespace['_add_to_vector_store'], hashlib.sha256(source.encode()).hexdigest()


class ExtractionLLM:
    def __init__(self, api):
        self.api = api

    def generate_response(self, messages, response_format=None, **kwargs):
        response = self.api.chat(messages, phase='extraction', json_mode=True)
        obj = json.loads(response)
        if not isinstance(obj.get('facts'), list) or not all(isinstance(x, str) and x for x in obj['facts']):
            raise ValueError('Invalid extracted facts')
        return response


def validate_program(actions, memories):
    if not isinstance(actions, list):
        raise ValueError('memory must be a list')
    ids = {str(m['id']) for m in memories}
    deleted = set()
    for action in actions:
        if (action.get('event') not in OPS or not isinstance(action.get('text'), str) or not action['text']
                or not isinstance(action.get('id'), str)):
            raise ValueError('Malformed operation or empty text')
        if action['event'] != 'ADD':
            target = str(action.get('id'))
            if target not in ids or target in deleted:
                raise ValueError('Invalid or already deleted target memory')
            if action['event'] == 'DELETE':
                deleted.add(target)


def executable_program(raw, memories):
    """Keep exactly the operations Mem0 can execute; audit skipped teacher entries."""
    if not isinstance(raw, list):
        raise ValueError('Teacher memory field must be a list')
    ids, deleted, actions, skipped = {str(m['id']) for m in memories}, set(), [], []
    for index, entry in enumerate(raw):
        action = dict(entry)
        event, target = action.get('event'), action.get('id')
        reason = None
        if not isinstance(action.get('text'), str) or not action['text']:
            reason = 'empty_text'
        elif event not in OPS:
            reason = 'unknown_event'
        elif event != 'ADD' and (target not in ids or target in deleted):
            reason = 'unresolvable_target'
        if reason:
            skipped.append(dict(action=action, reason=reason))
            continue
        if event == 'ADD':
            action['id'] = str(action.get('id', f'new_{index}'))
        elif event == 'DELETE':
            deleted.add(target)
        actions.append(action)
    return actions, skipped


class ErrorCapture(logging.Handler):
    def __init__(self, owner):
        super().__init__(logging.ERROR)
        self.owner = owner
        self.messages = []

    def emit(self, record):
        if _memory_context.get() == self.owner:
            self.messages.append(record.getMessage())


class DecisionMemory(Memory):
    _patched_vector_add, upstream_sha256 = patched_add()

    def _add_to_vector_store(self, *args, **kwargs):
        token = _memory_context.set(id(self))
        try:
            return self._patched_vector_add(*args, **kwargs)
        finally:
            _memory_context.reset(token)

    def __init__(self, encoder, api, directory, policy=None, native=False, reset_state=False):
        # Inject shared frozen services; all three arms use the same upstream store and methods.
        # Avoid repeated encoder loading and migration-store side effects per conversation.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config = MemoryConfig(custom_fact_extraction_prompt=FIXED_EXTRACTION_PROMPT)
        self.custom_fact_extraction_prompt = FIXED_EXTRACTION_PROMPT
        self.custom_update_memory_prompt = None
        self.embedding_model = encoder
        self.llm = ExtractionLLM(api)
        self.collection_name = 'memory'
        self.vector_store = Qdrant(collection_name='memory', embedding_model_dims=encoder.dim,
                                   client=QdrantClient(':memory:'))
        self.db = SQLiteManager(str(directory / 'history.sqlite'))
        self.api_version = 'v1.1'
        self.enable_graph, self.graph = False, None
        self.api, self.policy = api, policy
        self.native, self.reset_state = native, reset_state
        self.state = None
        self.traces = []
        self.observation_seconds = 0.0

    def ingest(self, messages, date, session_id):
        self.current_messages = deepcopy(messages)
        self.current_date, self.current_session_id = date, session_id
        start = time.perf_counter()
        if self.policy is not None and self.policy.stateful:
            device = next(self.policy.parameters()).device
            z = torch.from_numpy(self.embedding_model.encode_history(turn_texts(messages))).to(device)
            with torch.no_grad():
                self.state = self.policy.observe(z, None if self.reset_state else self.state)
        self.observation_seconds = time.perf_counter() - start
        errors = ErrorCapture(id(self))
        upstream.logger.addHandler(errors)
        before = len(self.traces)
        try:
            result = self.add(messages, user_id='subject', metadata={'session_date': date, 'session_id': session_id})
        finally:
            upstream.logger.removeHandler(errors)
        if errors.messages:
            raise RuntimeError('Mem0 suppressed an error: ' + '; '.join(errors.messages))
        if len(self.traces) == before:
            self.traces.append(dict(messages=messages, facts=[], memories=[], actions=[], mapping={},
                                    date=date, session_id=session_id, observation_seconds=self.observation_seconds,
                                    decision_seconds=0.0))
        trace = self.traces[-1]
        expected = [a['event'] for a in trace['actions'] if a['event'] != 'NONE']
        actual = [a['event'] for a in result['results']]
        if expected != actual:
            raise RuntimeError(f'Execution mismatch: {expected} != {actual}')
        trace['executed'] = result['results']
        trace['ingest_seconds'] = time.perf_counter() - start
        return result

    def decide_update(self, facts, memories, prompt, mapping):
        start = time.perf_counter()
        skipped = []
        if self.policy is None:
            response = self.api.chat([{'role': 'user', 'content': prompt}], 'decision', json_mode=True)
            native_actions, skipped = executable_program(json.loads(response).get('memory'), memories)
            decisions = [{k: a[k] for k in ('id', 'event')} for a in native_actions]
        else:
            device = next(self.policy.parameters()).device
            fact_z = torch.from_numpy(self.embedding_model.encode(facts)).to(device)
            mem_z = torch.from_numpy(self.embedding_model.encode([m['text'] for m in memories])).to(device)
            program = self.policy.decide(fact_z, mem_z, self.state,
                                         max_actions=max(128, 2 * (len(facts) + len(memories)) + 1))
            decisions = [dict(event=OPS[op], id=str(len(memories) + i) if op == 0 else memories[target - 1]['id'])
                         for i, (op, target) in enumerate(program)]
            native_actions = None
        elapsed = time.perf_counter() - start
        if self.native:
            if native_actions is None:
                raise ValueError('native applies only to the LLM reference')
            actions = native_actions
        else:
            actions = self.render(facts, memories, decisions)
        validate_program(actions, memories)
        # Validate that the exact recorded program can be used as a training label.
        labels(actions, memories)
        self.traces.append(dict(messages=self.current_messages, facts=facts, memories=deepcopy(memories),
                                mapping=deepcopy(mapping), actions=actions, native_actions=native_actions,
                                date=self.current_date, session_id=self.current_session_id,
                                skipped_teacher_actions=skipped,
                                observation_seconds=self.observation_seconds, decision_seconds=elapsed))
        return json.dumps({'memory': actions})

    def render(self, facts, memories, decisions):
        old = {str(m['id']): m['text'] for m in memories}
        writes = [dict(index=i, **a) for i, a in enumerate(decisions) if a['event'] in ('ADD', 'UPDATE')]
        texts = {}
        if writes:
            prompt = ('Realize the already fixed memory operations as concise standalone factual sentences. '
                      'Do not choose, remove, or add operations. For UPDATE preserve compatible old details '
                      'and replace contradicted details. For ADD express new information without duplicating '
                      'other writes. Use only the supplied facts and memories. Return JSON {"writes": '
                      '[{"index": integer, "text": string}]} with exactly one item per requested write.\n' +
                      json.dumps(dict(facts=facts, memories=memories, writes=writes), ensure_ascii=False))
            messages = [dict(role='user', content=prompt)]
            expected = {x['index'] for x in writes}
            for attempt in range(3):
                response = self.api.chat(messages, 'render', json_mode=True)
                value = json.loads(response)
                entries = value.get('writes', [])
                texts = {x.get('index'): x.get('text') for x in entries}
                if (len(entries) == len(writes) and set(texts) == expected and
                        all(isinstance(v, str) and v.strip() for v in texts.values())):
                    break
                messages += [dict(role='assistant', content=response), dict(role='user', content=
                    f'Invalid output. Return exactly {len(writes)} separate writes with indices '
                    f'{sorted(expected)}. Never merge writes. Copy no other indices. All texts must be nonempty.')]
            else:
                raise ValueError('Renderer changed the operation program after three attempts')
        return [dict(**a, text=texts[i] if i in texts else old[str(a['id'])],
                     **({'old_memory': old[str(a['id'])]} if a['event'] == 'UPDATE' else {}))
                for i, a in enumerate(decisions)]

    def close(self):
        self.vector_store.client.close()
        self.db.connection.close()
