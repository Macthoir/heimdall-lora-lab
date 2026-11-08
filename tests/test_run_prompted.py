"""Checks for scripts/run_prompted.py with mocked clients. Nothing here touches the network."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_prompted  # noqa: E402
from common import load_template  # noqa: E402

GOOD = '{"task_type":"debugging","complexity":"low"}'


class ApiError(Exception):
    def __init__(self, status, msg="boom"):
        super().__init__(msg)
        self.status_code = status


def anthropic_reply(text, stop="end_turn"):
    return NS(content=[NS(type="thinking", text=""), NS(type="text", text=text)],
              usage=NS(input_tokens=300, output_tokens=12), stop_reason=stop)


def openai_reply(text, finish="stop"):
    return NS(choices=[NS(message=NS(content=text), finish_reason=finish)],
              usage=NS(prompt_tokens=300, completion_tokens=12))


class FakeAnthropic:
    def __init__(self, script):
        self.script = list(script)  # each item: a reply text, or an exception to raise
        self.calls = []
        self.messages = NS(create=self.create)

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if self.script else GOOD
        if isinstance(item, Exception):
            raise item
        return anthropic_reply(item)


class FakeOpenAI:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0) if self.script else GOOD
        if isinstance(item, Exception):
            raise item
        return openai_reply(item)


class RunPrompted(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        rows = [{"id": f"v{i}", "prompt": f"prompt {i}", "task_type": "debugging", "complexity": "low"} for i in range(4)]
        (self.dir / "val.jsonl").write_text("".join(json.dumps(r) + "\r\n" for r in rows), encoding="utf-8", newline="")
        self.sleep = mock.patch.object(run_prompted.time, "sleep")
        self.sleeps = self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_main(self, extra, client=None, backend="anthropic"):
        argv = ["--backend", backend, "--model", "m", "--name", "t", "--data", str(self.dir / "val.jsonl"),
                "--results-dir", str(self.dir / "res"), *extra]
        if backend == "openai":
            argv += ["--base-url", "http://localhost:8000/v1"]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = run_prompted.main(argv, client)
        return code, out.getvalue(), err.getvalue()

    def preds(self):
        p = self.dir / "res" / "t_val_preds.jsonl"
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]

    def test_anthropic_request_shape_and_output_rows(self):
        client = FakeAnthropic([GOOD, "```json\n" + GOOD + "\n```", "not json", GOOD])
        code, _, err = self.run_main([], client)
        self.assertEqual(code, 0)
        call = client.calls[0]
        self.assertEqual(call["system"], load_template())
        self.assertEqual(call["messages"], [{"role": "user", "content": "prompt 0"}])
        self.assertNotIn("temperature", call)  # Opus 5.5 rejects sampling parameters
        self.assertEqual(call["output_config"], {"effort": "low"})
        self.assertEqual(call["max_tokens"], 2048)
        rows = self.preds()
        self.assertEqual([r["id"] for r in rows], ["v0", "v1", "v2", "v3"])
        self.assertEqual(set(rows[0]), {"id", "task_type", "complexity", "raw", "parse_ok"})
        self.assertEqual((rows[0]["task_type"], rows[0]["complexity"], rows[0]["parse_ok"]), ("debugging", "low", True))
        self.assertTrue(rows[1]["parse_ok"])  # fenced reply is accepted
        self.assertEqual((rows[2]["task_type"], rows[2]["complexity"], rows[2]["parse_ok"]), (None, None, False))
        self.assertEqual(rows[2]["raw"], "not json")
        self.assertIn("'requests': 4", err)
        self.assertIn("'input_tokens': 1200", err)

    def test_openai_request_shape(self):
        client = FakeOpenAI([])
        code, _, _ = self.run_main(["--limit", "2"], client, backend="openai")
        self.assertEqual(code, 0)
        call = client.calls[0]
        self.assertEqual(call["messages"][0], {"role": "system", "content": load_template()})
        self.assertEqual(call["messages"][1], {"role": "user", "content": "prompt 0"})
        self.assertEqual((call["temperature"], call["max_tokens"]), (0, 64))
        self.assertEqual(len(self.preds()), 2)

    def test_openai_backend_requires_base_url(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run_prompted.main(["--backend", "openai", "--model", "m", "--name", "t"])

    def test_resume_skips_written_ids(self):
        first = FakeAnthropic([])
        self.run_main(["--limit", "2"], first)
        second = FakeAnthropic([])
        code, _, _ = self.run_main([], second)
        self.assertEqual(code, 0)
        self.assertEqual([c["messages"][0]["content"] for c in second.calls], ["prompt 2", "prompt 3"])
        self.assertEqual([r["id"] for r in self.preds()], ["v0", "v1", "v2", "v3"])
        third = FakeAnthropic([])
        self.run_main([], third)
        self.assertEqual(third.calls, [])

    def test_resume_after_torn_last_line(self):
        res = self.dir / "res"
        res.mkdir()
        good = json.dumps({"id": "v0", "task_type": "debugging", "complexity": "low", "raw": GOOD, "parse_ok": True})
        (res / "t_val_preds.jsonl").write_text(good + '\n{"id": "v1", "task_ty', encoding="utf-8")
        client = FakeAnthropic([])
        self.run_main([], client)
        self.assertEqual(len(client.calls), 3)  # v1 (torn), v2, v3
        lines = (res / "t_val_preds.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 5)  # v0, the torn fragment, v1, v2, v3

    def test_stops_at_once_on_auth_billing_and_model_errors(self):
        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                res = self.dir / "res" / "t_val_preds.jsonl"
                res.unlink(missing_ok=True)
                client = FakeAnthropic([GOOD, ApiError(status)])
                code, _, err = self.run_main([], client)
                self.assertEqual(code, 1)
                self.assertEqual(len(client.calls), 2)  # no retry
                self.assertEqual(len(self.preds()), 1)  # the row before the error is kept
                self.assertIn("Stopped", err)
        self.sleeps.assert_not_called()

    def test_connection_errors_are_not_retried(self):
        client = FakeOpenAI([ConnectionError("refused")])
        code, _, _ = self.run_main([], client, backend="openai")
        self.assertEqual(code, 1)
        self.assertEqual(len(client.calls), 1)

    def test_retries_rate_limit_and_5xx_then_succeeds(self):
        client = FakeAnthropic([ApiError(429), ApiError(503), ApiError(529), GOOD])
        code, _, _ = self.run_main(["--limit", "1"], client)
        self.assertEqual(code, 0)
        self.assertEqual(len(client.calls), 4)
        self.assertEqual(self.sleeps.call_count, 3)
        self.assertEqual(len(self.preds()), 1)

    def test_gives_up_after_three_retries(self):
        client = FakeAnthropic([ApiError(500)] * 10)
        code, _, err = self.run_main(["--limit", "1"], client)
        self.assertEqual(code, 1)
        self.assertEqual(len(client.calls), 4)  # first try + 3 retries
        self.assertIn("Stopped", err)

    def test_dry_run_prints_first_request_and_sends_nothing(self):
        client = FakeAnthropic([])
        with mock.patch.object(run_prompted, "make_client") as make_client:
            code, out, _ = self.run_main(["--dry-run"], client)
        self.assertEqual(code, 0)
        make_client.assert_not_called()
        self.assertEqual(client.calls, [])
        self.assertFalse((self.dir / "res").exists())
        self.assertIn("nothing was sent", out)
        self.assertIn("prompt 0", out)
        self.assertIn("You label prompts", out)

    def test_dry_run_needs_no_sdk_or_key(self):
        # A fresh interpreter, with the SDKs blocked from importing.
        code = (
            "import sys; sys.modules['anthropic'] = None; sys.modules['openai'] = None; "
            "sys.path.insert(0, %r); import run_prompted; "
            "sys.exit(run_prompted.main(['--backend','anthropic','--model','m','--name','t','--dry-run','--data',%r]))"
        ) % (str(SCRIPTS), str(self.dir / "val.jsonl"))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("nothing was sent", r.stdout)

    def test_cut_off_replies_are_parse_failures_and_reported(self):
        client = FakeAnthropic([])
        client.create = lambda **kw: anthropic_reply("", stop="max_tokens")
        client.messages = NS(create=client.create)
        code, _, err = self.run_main(["--limit", "2"], client)
        self.assertEqual(code, 0)
        self.assertTrue(all(not r["parse_ok"] for r in self.preds()))
        self.assertIn("max_tokens", err)


if __name__ == "__main__":
    unittest.main()
