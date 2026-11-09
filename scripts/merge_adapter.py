"""Merge the LoRA adapter into the base model and save a standalone checkpoint.

    python scripts/merge_adapter.py
    python scripts/merge_adapter.py --adapter checkpoints/lora-v1/adapter --out checkpoints/lora-v1/merged

Loads Qwen/Qwen2.5-1.5B-Instruct in bf16 on CPU, attaches the PEFT adapter,
calls merge_and_unload() and writes the merged weights plus the tokenizer, so
the output directory loads with plain transformers (or vLLM / GGUF conversion)
with no adapter at serving time. Refuses to overwrite a non-empty --out unless
--overwrite is given.

Needs torch, transformers and peft. Paths are relative to the repo root.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "checkpoints/lora-v1/adapter"
DEFAULT_OUT = "checkpoints/lora-v1/merged"


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO / path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE_MODEL, help="base model id or local path (default: %(default)s)")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER, help="PEFT adapter dir (default: %(default)s)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true", help="write into --out even if it is not empty")
    args = ap.parse_args()

    adapter = resolve(args.adapter)
    out = resolve(args.out)

    if not (adapter / "adapter_config.json").is_file():
        print(f"error: no adapter_config.json in {adapter}", file=sys.stderr)
        return 1
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        print(f"error: {out} is not empty (use --overwrite)", file=sys.stderr)
        return 1

    # Imported here so --help works without the heavy dependencies installed.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base {args.base} (bf16, cpu)")
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.base)

    print(f"attaching adapter {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter))

    print("merge_and_unload()")
    model = model.merge_and_unload()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"saved merged model to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
