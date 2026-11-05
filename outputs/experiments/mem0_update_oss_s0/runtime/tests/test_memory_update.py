import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hi_ontop.factored_slot import FactoredSlot
from hi_ontop.memory_update.common import clean_session
from hi_ontop.memory_update.model import UpdatePolicy, labels


def test_streaming_matches_existing_factored_slot():
    torch.manual_seed(5)
    source = FactoredSlot(z_dim=12, slot_dim=8, dropout=0).eval()
    policy = UpdatePolicy(z_dim=12, width=16, slot_dim=8).eval()
    own = policy.state_dict()
    own.update({k: v for k, v in source.state_dict().items() if k in own})
    policy.load_state_dict(own)
    inputs = torch.randn(9, 12)
    whole = policy.observe(inputs)
    streamed = policy.observe(inputs[4:], policy.observe(inputs[:4]))
    torch.testing.assert_close(whole, source.encode_seq(inputs)[-1])
    torch.testing.assert_close(whole, streamed)


def test_empty_history_and_conversation_reset():
    model = UpdatePolicy(z_dim=12, width=16, slot_dim=8).eval()
    torch.testing.assert_close(model.observe(torch.empty(0, 12)), model.keys)
    a, b = torch.randn(6, 12), torch.randn(6, 12)
    before = model.observe(a)
    model.observe(b)
    torch.testing.assert_close(model.observe(a), before)
    assert not torch.allclose(model.observe(a, model.observe(b)), before)


def test_stateless_cannot_access_history():
    model = UpdatePolicy(z_dim=12, width=16, slot_dim=8, stateful=False)
    assert model.observe(torch.randn(5, 12)) is None
    f, m = torch.randn(3, 12), torch.randn(2, 12)
    torch.testing.assert_close(model.prepare(f, m, torch.randn(5, 8))[0], model.prepare(f, m, None)[0])


def test_capacity_control():
    counts = [sum(p.numel() for p in UpdatePolicy(stateful=s).parameters()) for s in (False, True)]
    assert abs(counts[0] - counts[1]) / counts[1] < 0.01


def test_pointer_candidates_are_equivariant():
    model = UpdatePolicy(z_dim=12, width=16, slot_dim=8).eval()
    f, m, s = torch.randn(3, 12), torch.randn(3, 12), model.observe(torch.randn(2, 12))
    h1, c1 = model.prepare(f, m, s)
    h2, c2 = model.prepare(f, m[[2, 0, 1]], s)
    _, op1, ptr1 = model.step(h1, c1, 4, 0)
    _, op2, ptr2 = model.step(h2, c2, 4, 0)
    torch.testing.assert_close(op1, op2)
    torch.testing.assert_close(ptr1[[0, 3, 1, 2]], ptr2)


def test_multi_action_labels_and_gradients():
    old = [dict(id='7', text='old'), dict(id='21', text='obsolete')]
    actions = [dict(id='8', event='ADD'), dict(id='7', event='UPDATE'), dict(id='21', event='DELETE')]
    program = labels(actions, old)
    assert program == [(0, 0), (1, 1), (2, 2)]
    model = UpdatePolicy(z_dim=12, width=16, slot_dim=8)
    loss, op, target = model.loss(torch.randn(2, 12), torch.randn(2, 12), model.observe(torch.randn(4, 12)), program)
    torch.testing.assert_close(loss.detach(), op + target)
    loss.backward()
    assert model.keys.grad.abs().sum() > 0
    assert model.pointer.weight.grad.abs().sum() > 0
    assert model.op_head.weight.grad.abs().sum() > 0
    assert not hasattr(model, 'change_heads')


def test_labels_reject_bad_targets():
    with pytest.raises(KeyError):
        labels([dict(id='unknown', event='UPDATE')], [dict(id='1')])
    assert labels([dict(id='1', event='UPDATE'), dict(id='1', event='DELETE')], [dict(id='1')]) == [(1, 1), (2, 1)]
    with pytest.raises(ValueError, match='deleted'):
        labels([dict(id='1', event='DELETE'), dict(id='1', event='UPDATE')], [dict(id='1')])


def test_only_executable_teacher_actions_become_labels():
    from hi_ontop.memory_update.adapter import executable_program
    raw = [dict(id='0', text='old', event='NONE'), dict(id='12', text='new', event='ADD'),
           dict(id='13', text='unresolvable', event='UPDATE')]
    actions, skipped = executable_program(raw, [dict(id='0', text='old')])
    assert actions == raw[:2]
    assert skipped == [dict(action=raw[2], reason='unresolvable_target')]


def test_annotation_filter():
    cleaned = clean_session([dict(role='user', content='Hello', has_answer=True, answer='secret')], '2024-01-01')
    assert set(cleaned[0]) == {'role', 'content'}
    assert 'secret' not in str(cleaned) and 'has_answer' not in str(cleaned)


def test_empty_memory_loss_is_finite():
    model = UpdatePolicy(z_dim=12, width=16, slot_dim=8, stateful=False)
    loss, _, _ = model.loss(torch.randn(1, 12), torch.empty(0, 12), None, [(0, 0)])
    assert torch.isfinite(loss)


class FakeEncoder:
    dim = 12
    config = SimpleNamespace(embedding_dims=12)

    def embed(self, text, memory_action=None):
        return np.ones(12).tolist()


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.round = 0

    def chat(self, messages, phase, **kwargs):
        self.calls.append(phase)
        if phase == 'extraction':
            self.round += 1
            return json.dumps({'facts': ['Lives in Rome' if self.round == 1 else 'Lives in Paris']})
        if phase == 'decision':
            return json.dumps({'memory': [dict(id='0', event='ADD' if self.round == 1 else 'UPDATE', text='native wording')]})
        if phase == 'render':
            return json.dumps({'writes': [dict(index=0, text='Lives in Rome' if self.round == 1 else 'Lives in Paris')]})
        raise AssertionError(phase)


def test_real_mem0_executes_hook_and_common_render(tmp_path):
    from hi_ontop.memory_update.adapter import DecisionMemory
    api = FakeAPI()
    memory = DecisionMemory(FakeEncoder(), api, tmp_path)
    try:
        first = memory.ingest([dict(role='user', content='I live in Rome')], '2024-01-01', 's1')
        second = memory.ingest([dict(role='user', content='I moved to Paris')], '2024-03-01', 's2')
        assert first['results'][0]['event'] == 'ADD'
        assert second['results'][0]['event'] == 'UPDATE'
        assert first['results'][0]['id'] == second['results'][0]['id']
        found = memory.search('Where do I live?', user_id='subject')['results']
        assert found[0]['memory'] == 'Lives in Paris'
        assert api.calls == ['extraction', 'decision', 'render'] * 2
    finally:
        memory.close()


def test_upstream_swallowed_errors_fail_experiment(tmp_path):
    from hi_ontop.memory_update.adapter import DecisionMemory
    api = FakeAPI()
    api.chat = lambda *a, **k: '{"bad": "response"}'
    memory = DecisionMemory(FakeEncoder(), api, tmp_path)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            memory.ingest([dict(role='user', content='Hello')], '2024-01-01', 's1')
    finally:
        memory.close()


def test_render_retries_without_changing_decisions(tmp_path):
    from hi_ontop.memory_update.adapter import DecisionMemory
    api = FakeAPI()
    responses = iter(['{"writes":[{"index":0,"text":"merged"}]}',
                      '{"writes":[{"index":0,"text":"A"},{"index":1,"text":"B"}]}'])
    api.chat = lambda *a, **k: next(responses)
    memory = DecisionMemory(FakeEncoder(), api, tmp_path)
    try:
        decisions = [dict(id='0', event='ADD'), dict(id='1', event='ADD')]
        rendered = memory.render(['A', 'B'], [], decisions)
        assert [(a['id'], a['event'], a['text']) for a in rendered] == [('0', 'ADD', 'A'), ('1', 'ADD', 'B')]
    finally:
        memory.close()


def test_native_storage_matches_unmodified_mem0(tmp_path):
    from types import MethodType
    from mem0 import Memory
    from hi_ontop.memory_update.adapter import DecisionMemory
    hooked_api, native_api = FakeAPI(), FakeAPI()
    hooked = DecisionMemory(FakeEncoder(), hooked_api, tmp_path / 'hooked', native=True)
    original = DecisionMemory(FakeEncoder(), native_api, tmp_path / 'original')
    original._add_to_vector_store = MethodType(Memory._add_to_vector_store, original)

    class NativeLLM:
        def generate_response(self, messages, **kwargs):
            phase = 'extraction' if messages[0]['role'] == 'system' else 'decision'
            return native_api.chat(messages, phase=phase)

    original.llm = NativeLLM()
    try:
        for i, text in enumerate(['I live in Rome', 'I moved to Paris']):
            messages = [dict(role='user', content=text)]
            a = hooked.ingest(messages, '2024-01-01', str(i))
            b = original.add(messages, user_id='subject')
            assert [(x['event'], x['memory']) for x in a['results']] == [(x['event'], x['memory']) for x in b['results']]
        assert [x['memory'] for x in hooked.search('home', user_id='subject')['results']] == [x['memory'] for x in original.search('home', user_id='subject')['results']]
    finally:
        hooked.close()
        original.close()


def test_job_capacity_counts_queued_gpu_reservations():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/mem0_update/vessl_train.py'
    spec = importlib.util.spec_from_file_location('vessl_train', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.capacity([dict(workloadState='running', resourceSpecAcceleratorCount=1),
                            dict(workloadState='pending', resourceSpecAcceleratorCount=1),
                            dict(workloadState='succeeded', resourceSpecAcceleratorCount=4)]) == (2, 2)
    with pytest.raises(KeyError):
        module.capacity([dict(workloadState='starting')])


def test_concurrent_memories_do_not_capture_each_others_errors(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from hi_ontop.memory_update.adapter import DecisionMemory, upstream
    barrier = threading.Barrier(2)

    class ConcurrentAPI(FakeAPI):
        def __init__(self, fail):
            super().__init__()
            self.fail = fail

        def chat(self, messages, phase, **kwargs):
            if phase == 'extraction':
                barrier.wait(timeout=10)
                if self.fail:
                    upstream.logger.error('fixture error in one conversation')
                barrier.wait(timeout=10)
            return super().chat(messages, phase, **kwargs)

    def run(fail):
        memory = DecisionMemory(FakeEncoder(), ConcurrentAPI(fail), tmp_path / str(fail))
        try:
            if fail:
                with pytest.raises(RuntimeError, match='fixture error'):
                    memory.ingest([dict(role='user', content='I live in Rome')], '2024-01-01', 's1')
            else:
                assert memory.ingest([dict(role='user', content='I live in Rome')], '2024-01-01', 's1')['results']
        finally:
            memory.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [True, False]))


def test_api_forks_separate_usage_but_share_cache_and_transport():
    from hi_ontop.memory_update.common import API
    root = object.__new__(API)
    root.records, root.client, root.db, root.lock = [], object(), object(), object()
    a, b = root.fork(), root.fork()
    a.records.append({'phase': 'decision'})
    assert not b.records and not root.records
    assert a.client is b.client is root.client
    assert a.db is b.db is root.db
    assert a.lock is b.lock is root.lock


def test_bounded_workers_does_not_eagerly_load_dataset():
    import threading
    from hi_ontop.memory_update.common import bounded_workers
    lock = threading.Lock()
    submitted, completed = [], []
    def items():
        for i in range(50):
            with lock:
                assert len(submitted) - len(completed) < 4
                submitted.append(i)
            yield i
    def consume(i):
        with lock:
            completed.append(i)
    bounded_workers(consume, items(), 4)
    assert sorted(completed) == list(range(50))
