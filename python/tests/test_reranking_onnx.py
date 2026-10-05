"""The cross-encoder's forward pass on ONNX Runtime: same preprocessing, same scores, torch as the fallback.

These run without a network or the model download. The real model is checked twice elsewhere: at load
(onnx_matches_torch on fixed pairs, else torch) and once by measurement (63 SPARK queries, same order 63/63).
"""
import numpy as np
import pytest
import torch

from src.server.services.search import reranking_strategy as rs


class _Tok:
    def __init__(self):
        self.calls = []

    def __call__(self, a, b, **kw):
        self.calls.append(kw)
        n = len(a)
        return {"input_ids": np.ones((n, 4), dtype=np.int32), "attention_mask": np.ones((n, 4), dtype=np.int32),
                "token_type_ids": np.zeros((n, 4), dtype=np.int32), "extra": np.zeros((n, 1))}


class _CE:
    """The parts of a CrossEncoder the ONNX path borrows."""

    max_length = 512

    def __init__(self, scores=None):
        self.tokenizer = _Tok()
        self.activation_fn = torch.nn.Sigmoid()
        self._scores = scores

    def predict(self, pairs, **_):
        return np.asarray(self._scores if self._scores is not None else [0.9, 0.1, 0.5][: len(pairs)])


class _Session:
    def __init__(self, logits):
        self.fed = None
        self._logits = logits

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, _, feed):
        self.fed = feed
        return [np.asarray(self._logits, dtype=np.float32).reshape(-1, 1)]


def _onnx(ce, logits):
    m = rs.OnnxCrossEncoder.__new__(rs.OnnxCrossEncoder)
    m._np, m._ce, m._session = np, ce, _Session(logits)
    m._inputs = [i.name for i in m._session.get_inputs()]
    return m


def test_predict_uses_the_cross_encoders_own_preprocessing_and_activation():
    ce = _CE()
    m = _onnx(ce, [0.0, 2.0])
    out = m.predict([["q", "a"], ("q", "b")])
    assert out.shape == (2,)
    assert np.allclose(out, torch.sigmoid(torch.tensor([0.0, 2.0])).numpy())       # the CE's activation applied
    assert ce.tokenizer.calls[0] == {"padding": True, "truncation": True, "max_length": 512, "return_tensors": "np"}
    assert set(m._session.fed) == {"input_ids", "attention_mask", "token_type_ids"}  # only the model's inputs
    assert all(v.dtype == np.int64 for v in m._session.fed.values())


def test_no_pairs_no_call():
    m = _onnx(_CE(), [])
    assert m.predict([]).shape == (0,) and m._session.fed is None


def test_the_self_check_accepts_only_the_same_ranking():
    torch_model = _CE(scores=[0.9, 0.1, 0.5])
    assert rs.onnx_matches_torch(_CE(scores=[0.9001, 0.1, 0.5]), torch_model)
    assert not rs.onnx_matches_torch(_CE(scores=[0.4, 0.1, 0.5]), torch_model)       # order changed
    assert not rs.onnx_matches_torch(_CE(scores=[0.9, 0.1, 0.51]), torch_model)      # score off by 1e-2


@pytest.mark.parametrize("backend,onnx_ok,matches,expect_onnx", [
    ("onnx", True, True, True),
    ("onnx", False, True, False),     # ONNX cannot load: torch, loudly
    ("onnx", True, False, False),     # ONNX loads but ranks differently: torch, loudly
    ("torch", True, True, False),     # switched off by configuration
])
def test_loading_falls_back_to_torch_unless_onnx_is_proven(monkeypatch, backend, onnx_ok, matches, expect_onnx):
    torch_model = _CE()
    monkeypatch.setattr(rs, "CrossEncoder", lambda name: torch_model)
    monkeypatch.setattr(rs, "CROSSENCODER_AVAILABLE", True)
    monkeypatch.setattr(rs, "RERANK_BACKEND", backend)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f: "/tmp/model.onnx")

    class _Fake:
        def __init__(self, ce, path, threads=2):
            if not onnx_ok:
                raise ImportError("No module named 'onnxruntime'")

        def predict(self, pairs, **_):
            return np.asarray([0.9, 0.1, 0.5]) if matches else np.asarray([0.1, 0.9, 0.5])
    monkeypatch.setattr(rs, "OnnxCrossEncoder", _Fake)
    model = rs.RerankingStrategy().model
    assert isinstance(model, _Fake) is expect_onnx
    assert (model is torch_model) is (not expect_onnx)


# ── The GPU reranker on the DGX first, the local model as the fallback ──────────────────────────────

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _Client:
    def __init__(self, behave):
        self.behave, self.calls = behave, 0

    def post(self, url, json):
        self.calls += 1
        return self.behave(json)


def _remote(behave, local=None, cooldown=30.0):
    client = _Client(behave)
    r = rs.RemoteFirstReranker("http://dgx", rs.DEFAULT_RERANKING_MODEL, local or _CE(), 0.5, cooldown, client=client)
    return r, client


def _ok(scores):
    return lambda body: _Resp(200, {"model": body["model"], "scores": scores[: len(body["pairs"])]})


def test_the_gpu_answers_first():
    r, c = _remote(_ok([0.3, 0.7, 0.1]))
    assert np.allclose(r.predict([("q", "a"), ("q", "b"), ("q", "c")]), [0.3, 0.7, 0.1])
    assert r.last_path == "remote" and c.calls == 1


@pytest.mark.parametrize("behave", [
    lambda body: (_ for _ in ()).throw(TimeoutError("read timeout")),                  # DGX blip
    lambda body: _Resp(503, {}),                                                         # server error
    lambda body: _Resp(200, {"model": body["model"], "scores": [0.1]}),                 # wrong count
    lambda body: _Resp(200, {"model": "another-model", "scores": [0.1, 0.2, 0.3]}),     # wrong model
])
def test_any_remote_problem_falls_back_to_the_local_model_and_cools_down(behave):
    r, c = _remote(behave, local=_CE(scores=[0.9, 0.1, 0.5]))
    out = r.predict([("q", "a"), ("q", "b"), ("q", "c")])
    assert r.last_path == "local" and np.allclose(out, [0.9, 0.1, 0.5])
    r.predict([("q", "a"), ("q", "b"), ("q", "c")])
    assert c.calls == 1                       # skipped during the cooldown: one timeout per window, not per query


def test_a_stale_keepalive_connection_is_retried_once_not_treated_as_down():
    import httpx
    state = {"n": 0}

    def behave(body):
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return _Resp(200, {"model": body["model"], "scores": [0.4, 0.6]})
    r, c = _remote(behave)
    assert np.allclose(r.predict([("q", "a"), ("q", "b")]), [0.4, 0.6])
    assert r.last_path == "remote" and c.calls == 2
    r.predict([("q", "a"), ("q", "b")])
    assert c.calls == 3                       # no cooldown was started


def test_a_stale_connection_twice_falls_back():
    import httpx
    r, c = _remote(lambda body: (_ for _ in ()).throw(httpx.RemoteProtocolError("disconnected")),
                   local=_CE(scores=[0.9, 0.1]))
    assert np.allclose(r.predict([("q", "a"), ("q", "b")]), [0.9, 0.1])
    assert r.last_path == "local" and c.calls == 2


def test_after_the_cooldown_the_gpu_is_tried_again():
    state = {"up": False}
    r, c = _remote(lambda body: _Resp(200, {"model": body["model"], "scores": [0.5]}) if state["up"]
                   else _Resp(503, {}), cooldown=0.0)
    r.predict([("q", "a")])
    state["up"] = True
    r.predict([("q", "a")])
    assert r.last_path == "remote" and c.calls == 2


@pytest.mark.parametrize("url,reply,expect_remote", [
    ("", None, False),                                               # not configured: local only
    ("http://dgx", "same", True),                                    # ranks like local: used
    ("http://dgx", "different", False),                             # ranks differently: never used
    ("http://dgx", "down", True),                                    # down at load: tried per call
])
def test_the_remote_is_put_in_front_only_when_it_ranks_like_local(monkeypatch, url, reply, expect_remote):
    monkeypatch.setattr(rs, "RERANK_REMOTE_URL", url)
    local = _CE(scores=[0.9, 0.1, 0.5])
    strat = rs.RerankingStrategy.from_model(local, model_name=rs.DEFAULT_RERANKING_MODEL)

    def fake_scores(self, pairs):
        if reply == "down":
            raise ConnectionError("refused")
        return np.asarray([0.9, 0.1, 0.5] if reply == "same" else [0.1, 0.9, 0.5])
    monkeypatch.setattr(rs.RemoteFirstReranker, "remote_scores", fake_scores)
    out = strat._with_remote(local)
    assert isinstance(out, rs.RemoteFirstReranker) is expect_remote
    if not expect_remote:
        assert out is local


def test_the_warm_up_exercises_the_model(monkeypatch):
    from src.server.services.search import rag_service

    calls = []
    fake = type("S", (), {"model": type("M", (), {"predict": lambda self, p: calls.append(p)})()})()
    monkeypatch.setattr(rag_service, "_SHARED_RERANKER", fake)
    rag_service.warm_reranker()
    assert len(calls) == 1
