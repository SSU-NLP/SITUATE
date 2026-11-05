from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np


def bounded_workers(function, items, workers):
    """Keep only one active item per worker; propagate failures before scheduling more."""
    if workers < 1:
        raise ValueError('workers must be positive')
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        try:
            for _ in range(workers):
                item = next(iterator, None)
                if item is None:
                    break
                pending.add(pool.submit(function, item))
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                for _ in done:
                    item = next(iterator, None)
                    if item is None:
                        break
                    pending.add(pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


class API:
    def __init__(self, env_file, cache_path, model, extra_body=None):
        from dotenv import dotenv_values
        from openai import OpenAI
        env = {**os.environ, **dotenv_values(env_file)}
        headers = {k.replace('_', '-'): env[k] for k in
                   ('CF_ACCESS_CLIENT_ID', 'CF_ACCESS_CLIENT_SECRET') if env.get(k)}
        self.client = OpenAI(api_key=env['OPENAI_API_KEY'], base_url=env['OPENAI_BASE_URL'],
                             default_headers=headers, timeout=120, max_retries=3)
        self.model = model
        self.extra_body = extra_body
        self.db = sqlite3.connect(cache_path, timeout=120, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS api_calls (id INTEGER PRIMARY KEY, key TEXT, phase TEXT, value TEXT)')
        self.db.execute("INSERT INTO api_calls (key,phase,value) SELECT key,'legacy_cache_migration',value FROM responses r WHERE NOT EXISTS (SELECT 1 FROM api_calls c WHERE c.key=r.key)")
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
        endpoint = str(self.client.base_url)
        previous = self.db.execute("SELECT value FROM metadata WHERE key='endpoint'").fetchone()
        if previous and previous[0] != endpoint:
            raise ValueError('API endpoint changed; use a separate cache namespace')
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('endpoint',?)", (endpoint,))
        self.db.commit()
        self.lock = threading.Lock()
        self.records = []

    def fork(self):
        """Share the connection pool/cache, but keep each conversation's usage separate."""
        scoped = object.__new__(type(self))
        scoped.__dict__ = {**self.__dict__, 'records': []}
        return scoped

    def chat(self, messages, phase, json_mode=False, model=None, max_tokens=4096, use_cache=True):
        request = dict(model=model or self.model, messages=messages, temperature=0,
                       max_tokens=max_tokens)
        if self.extra_body and request['model'] == self.model:
            request['extra_body'] = self.extra_body
        if json_mode:
            request['response_format'] = {'type': 'json_object'}
        key = digest(request)
        with self.lock:
            row = self.db.execute('SELECT value FROM responses WHERE key=?', (key,)).fetchone() if use_cache else None
        if row:
            value = json.loads(row[0])
            cache_hit = True
        else:
            start = time.perf_counter()
            response = self.client.chat.completions.create(**request)
            elapsed = time.perf_counter() - start
            value = dict(text=response.choices[0].message.content, seconds=elapsed,
                         usage=response.usage.model_dump(), model=response.model)
            with self.lock:
                self.db.execute('INSERT INTO api_calls (key,phase,value) VALUES (?,?,?)', (key, phase, json.dumps(value)))
                self.db.commit()
            if response.choices[0].finish_reason != 'stop':
                raise RuntimeError(f"Incomplete API response: {response.choices[0].finish_reason}")
            if json_mode:
                json.loads(value['text'])
            with self.lock:
                if use_cache:
                    self.db.execute('INSERT OR IGNORE INTO responses VALUES (?, ?)', (key, json.dumps(value)))
                self.db.commit()
            cache_hit = False
        self.records.append(dict(phase=phase, key=key, cache_hit=cache_hit,
                                 **{k: v for k, v in value.items() if k != 'text'}))
        return value['text']


class Encoder:
    def __init__(self, model, revision, cache_path, device='cpu'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, revision=revision, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.db = sqlite3.connect(cache_path, timeout=120, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value BLOB)')
        self.lock = threading.Lock()
        self.namespace = (model, revision)
        from types import SimpleNamespace
        self.config = SimpleNamespace(embedding_dims=self.dim)

    def encode(self, texts):
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        keys = [digest((self.namespace, t)) for t in texts]
        with self.lock:
            rows = [self.db.execute('SELECT value FROM embeddings WHERE key=?', (k,)).fetchone() for k in keys]
            missing = [i for i, row in enumerate(rows) if row is None]
            if missing:
                vectors = self.model.encode([texts[i] for i in missing], batch_size=32,
                                             normalize_embeddings=True, show_progress_bar=False)
                for i, vector in zip(missing, vectors):
                    value = np.asarray(vector, dtype=np.float32).tobytes()
                    self.db.execute('INSERT OR IGNORE INTO embeddings VALUES (?, ?)', (keys[i], value))
                    rows[i] = (value,)
                self.db.commit()
        return np.stack([np.frombuffer(row[0], dtype=np.float32).copy() for row in rows])

    def embed(self, text, memory_action=None):
        return self.encode([text])[0].tolist()

    def history_chunks(self, texts):
        # Preserve long turns in causal chunks instead of silently dropping their tails.
        tokenizer = self.model.tokenizer
        width = self.model.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        chunks = []
        with self.lock:
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                if len(ids) <= width:
                    chunks.append(text)
                else:
                    chunks.extend(tokenizer.decode(ids[i:i + width], skip_special_tokens=True)
                                  for i in range(0, len(ids), width))
        return chunks

    def encode_history(self, texts):
        return self.encode(self.history_chunks(texts))


def clean_session(turns, date):
    # Evidence annotations and questions never enter the memory pipeline.
    return [dict(role=t['role'], content=f"[Session date: {date}]\n{t['content']}") for t in turns]


def turn_texts(turns):
    return [f"{t['role']}: {t['content']}" for t in turns]
