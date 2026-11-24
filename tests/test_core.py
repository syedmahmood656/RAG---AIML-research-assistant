"""
Unit tests that need NO model download, NO network and NO FAISS.
Run from the project root:   python -m unittest discover -s tests -v
"""

import asyncio
import os
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "frontend"))
os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
os.environ.pop("HF_TOKEN", None)

try:  # keep the tests runnable on a bare Python install
    import langchain_core.embeddings  # noqa: F401
except ImportError:
    pkg = types.ModuleType("langchain_core")
    mod = types.ModuleType("langchain_core.embeddings")
    mod.Embeddings = object
    pkg.embeddings = mod
    sys.modules["langchain_core"] = pkg
    sys.modules["langchain_core.embeddings"] = mod

import config  # noqa: E402
import rag  # noqa: E402
import sse  # noqa: E402
from sse_client import BackendError, decode_events  # noqa: E402


class Doc:
    def __init__(self, text, **meta):
        self.page_content, self.metadata = text, meta


def collect(agen):
    async def run():
        return [x async for x in agen]
    return asyncio.run(run())


class SSETests(unittest.TestCase):
    def roundtrip(self, messages):
        wire = "".join(sse.encode(e, p) for e, p in messages)
        return list(decode_events(wire.split("\n")))

    def test_newlines_quotes_unicode_survive(self):
        tricky = "line1\nline2 \"quoted\" \\ backslash\n\n- bullet ✓ 日本語\r\nend"
        out = self.roundtrip([("token", tricky), ("done", {"mode": "llm"})])
        self.assertEqual(out[0], ("token", tricky))
        self.assertEqual(out[1], ("done", {"mode": "llm"}))

    def test_multiple_tokens_in_order(self):
        out = self.roundtrip([("token", "a"), ("token", " b"), ("token", "\n")])
        self.assertEqual("".join(p for _, p in out), "a b\n")

    def test_error_event_raises(self):
        with self.assertRaises(BackendError):
            self.roundtrip([("sources", {"sources": []}), ("error", {"message": "boom"})])


class RetrievalHelpers(unittest.TestCase):
    def test_dedupe_keeps_best_ranked_and_limits_k(self):
        a, a_dup, b, c = Doc("Alpha  text"), Doc("alpha text"), Doc("Beta"), Doc("Gamma")
        out = rag.dedupe_results([(a, 0.1), (a_dup, 0.2), (b, 0.3), (c, 0.4)], k=2)
        self.assertEqual([d for d, _ in out], [a, b])

    def test_similarity_conversion(self):
        self.assertAlmostEqual(rag.l2_to_similarity(0.0), 1.0)
        self.assertAlmostEqual(rag.l2_to_similarity(1.0), 0.5)
        self.assertEqual(rag.l2_to_similarity(5.0), 0.0)

    def test_format_sources_handles_missing_metadata(self):
        src = rag.format_sources([(Doc("x" * 2000, seq_num=7), 0.4)])[0]
        self.assertEqual(src["title"], "Untitled excerpt")
        self.assertFalse(src["has_metadata"])
        self.assertEqual(len(src["snippet"]), 900)
        self.assertEqual(src["score"], 0.8)

    def test_build_messages_numbers_the_context(self):
        msgs = rag.build_messages("Q?", [(Doc("ctx one", Title="Paper A"), 0.1), (Doc("ctx two"), 0.2)])
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("[1] Paper A", msgs[1]["content"])
        self.assertIn("[2]\nctx two", msgs[1]["content"])
        self.assertTrue(msgs[1]["content"].rstrip().endswith("Question: Q?"))


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.results = [(Doc("some retrieved text", Title="Paper A"), 0.2)]
        self._saved = (rag._CLIENT, rag._LLM_ERROR)
        rag._CLIENT, rag._LLM_ERROR = None, None

    def tearDown(self):
        rag._CLIENT, rag._LLM_ERROR = self._saved

    @staticmethod
    def chunk(text):
        delta = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])

    def test_no_token_falls_back_to_extractive(self):
        events = collect(rag.stream_answer("q", self.results))  # token env var removed above
        self.assertEqual(events[0][0], "notice")
        self.assertIn("Paper A", events[1][1])

    def test_llm_tokens_are_streamed(self):
        chunks = [self.chunk("Hel"), self.chunk("lo"), self.chunk(None)]
        rag._CLIENT = types.SimpleNamespace(chat_completion=lambda **kw: iter(chunks))
        events = collect(rag.stream_answer("q", self.results))
        self.assertEqual(events, [("token", "Hel"), ("token", "lo")])

    def test_llm_failure_falls_back(self):
        def boom(**kw):
            raise RuntimeError("429 rate limit")
        rag._CLIENT = types.SimpleNamespace(chat_completion=boom)
        events = collect(rag.stream_answer("q", self.results))
        self.assertEqual(events[0][0], "notice")
        self.assertIn("rate limited", events[0][1])
        self.assertEqual(events[1][0], "token")


class ConfigTests(unittest.TestCase):
    def test_placeholder_tokens_are_ignored(self):
        for raw in (None, "", "  ", "API KEY", "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", '"your_token_here"'):
            self.assertIsNone(config._clean_token(raw), raw)
        self.assertEqual(config._clean_token(' "hf_realToken123" '), "hf_realToken123")


if __name__ == "__main__":
    unittest.main()
