"""Run a prompted model over the val rows and write predictions for score.py.

    # Opus 5.5 (key from ANTHROPIC_API_KEY or an `ant auth login` profile; paid)
    python scripts/run_prompted.py --backend anthropic --model claude-opus-5-5 --name opus

    # vLLM OpenAI-compatible server on the GPU box: prompted base model, then the LoRA adapter
    python scripts/run_prompted.py --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-1.5B-Instruct --name qwen_base
    python scripts/run_prompted.py --backend openai --base-url http://localhost:8000/v1 --model <adapter name> --name lora

Every request is shared/template_v1.txt as the system prompt and the raw prompt as the
user turn. Output: results/<name>_val_preds.jsonl, rows of
{"id", "task_type", "complexity", "raw", "parse_ok"}. Rerunning the same command
resumes: ids already in the file are skipped. --dry-run prints the first request
and sends nothing.

Request settings differ by backend:
- openai: temperature 0, max_tokens 64.
- anthropic: no temperature (Opus 5.5 returns a 400 for sampling parameters),
  output_config effort (default low), max_tokens 2048. Thinking is always on for
  Opus 5.5 and counts toward max_tokens, so 64 would cut replies off.

Stops at once on auth, billing, bad-model or any other non-retryable API error.
Retries only 429 and 5xx, 3 times. The SDKs are imported inside the backend that
needs them, so a dry run needs neither.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_template, parse_output, read_jsonl  # noqa: E402

MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = {"openai": 64, "anthropic": 2048}
USAGE = {"requests": 0, "input_tokens": 0, "output_tokens": 0}


class StopRun(Exception):
    """Stop cleanly; rows already written are kept and the same command resumes."""


def retryable(e: Exception) -> bool:
    # Only rate limits (429) and server errors (5xx). No status (connection refused,
    # timeout) is not retried: against a local vLLM it usually means the server is down.
    status = getattr(e, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


def build_request(args, template: str, prompt: str) -> dict:
    """The request body for one prompt, as plain data (also what --dry-run prints)."""
    if args.backend == "anthropic":
        return {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "system": template,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": args.effort},
        }
    return {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "messages": [{"role": "system", "content": template}, {"role": "user", "content": prompt}],
    }


def make_client(args):
    # max_retries=0: the SDKs would otherwise retry connection errors too; our policy decides.
    if args.backend == "anthropic":
        import anthropic  # lazy: only this backend needs the SDK
        return anthropic.Anthropic(max_retries=0)
    import openai  # lazy
    return openai.OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", max_retries=0)


def send(client, backend: str, request: dict):
    """One API call. Returns (reply text, input tokens, output tokens, stop reason)."""
    if backend == "anthropic":
        resp = client.messages.create(**request)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens, resp.stop_reason
    resp = client.chat.completions.create(**request)
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    return (choice.message.content or ""), getattr(usage, "prompt_tokens", 0) or 0, \
        getattr(usage, "completion_tokens", 0) or 0, choice.finish_reason


def call_with_retries(client, backend: str, request: dict):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return send(client, backend, request)
        except Exception as e:
            if not retryable(e) or attempt == MAX_RETRIES:
                raise StopRun(f"API error, not retrying: {e.__class__.__name__}: {e}") from e
            wait = 2 ** attempt * 5
            print(f"  API error ({e.__class__.__name__}), retry {attempt + 1}/{MAX_RETRIES} in {wait}s", file=sys.stderr)
            time.sleep(wait)


def load_done(path: Path) -> set:
    """Ids already written. A torn last line from a crash is ignored, not fatal."""
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (ValueError, KeyError, TypeError):
                print(f"warning: ignoring an unreadable line in {path}", file=sys.stderr)
    return done


def run(args, client=None) -> int:
    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        sys.exit("no rows to run")
    template = load_template()

    if args.dry_run:
        req = build_request(args, template, rows[0]["prompt"])
        print(f"backend: {args.backend}   base_url: {args.base_url}   rows: {len(rows)}   id: {rows[0]['id']}")
        print("--- request (nothing was sent) ---")
        print(json.dumps(req, indent=2, ensure_ascii=False))
        return 0

    out_path = Path(args.results_dir) / f"{args.name}_val_preds.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} rows, {len(done)} already in {out_path}, {len(todo)} to run", file=sys.stderr)
    if not todo:
        return 0

    if client is None:
        try:
            client = make_client(args)
        except Exception as e:
            print(f"Stopped: could not create the {args.backend} client: {e}", file=sys.stderr)
            return 1

    stops = {}
    USAGE.update(requests=0, input_tokens=0, output_tokens=0)
    torn = out_path.exists() and out_path.stat().st_size > 0 and not out_path.read_bytes().endswith(b"\n")
    with open(out_path, "a", encoding="utf-8", newline="\n") as out:
        if torn:
            out.write("\n")  # finish a line cut short by a crash
        try:
            for i, r in enumerate(todo, 1):
                text, n_in, n_out, stop = call_with_retries(client, args.backend, build_request(args, template, r["prompt"]))
                USAGE["requests"] += 1
                USAGE["input_tokens"] += n_in
                USAGE["output_tokens"] += n_out
                if stop in ("max_tokens", "length", "refusal"):
                    stops[stop] = stops.get(stop, 0) + 1
                label, ok = parse_output(text)
                out.write(json.dumps({
                    "id": r["id"],
                    "task_type": label["task_type"] if ok else None,
                    "complexity": label["complexity"] if ok else None,
                    "raw": text,
                    "parse_ok": ok,
                }) + "\n")
                out.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", file=sys.stderr)
        except StopRun as e:
            print(f"\nStopped: {e}", file=sys.stderr)
            print("Rows written so far are kept. Rerun the same command to resume.", file=sys.stderr)
            return 1
        finally:
            print(f"\nUsage: {USAGE}", file=sys.stderr)
            if stops:
                print(f"Cut-off or refused replies (counted as parse failures): {stops}", file=sys.stderr)
    return 0


def main(argv=None, client=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["anthropic", "openai"], required=True)
    ap.add_argument("--model", required=True, help="model id, or the LoRA adapter name on the vLLM server")
    ap.add_argument("--base-url", help="OpenAI-compatible server, e.g. http://localhost:8000/v1 (required for openai)")
    ap.add_argument("--name", required=True, help="writes results/<name>_val_preds.jsonl")
    ap.add_argument("--limit", type=int, help="only the first N rows of the data file")
    ap.add_argument("--dry-run", action="store_true", help="print the first request and send nothing")
    ap.add_argument("--data", default=str(REPO / "data" / "val.jsonl"))
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--max-tokens", type=int, help="default 64 (openai) or 2048 (anthropic)")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                    help="anthropic backend only")
    args = ap.parse_args(argv)
    if args.backend == "openai" and not args.base_url:
        ap.error("--backend openai needs --base-url (a vLLM server); this script never calls the paid OpenAI API")
    if args.max_tokens is None:
        args.max_tokens = DEFAULT_MAX_TOKENS[args.backend]
    return run(args, client)


if __name__ == "__main__":
    sys.exit(main())
