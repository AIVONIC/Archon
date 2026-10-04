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
