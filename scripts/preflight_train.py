"""Checks to run before paying for GPU time with train/lora_qwen25_1p5b.yaml.

    python scripts/preflight_train.py              # no GPU: YAML parses, data files load
    python scripts/preflight_train.py --tokenizer  # on the pod: token lengths + loss mask

Without --tokenizer it needs only PyYAML (a dependency of nemo-automodel). It checks that
the YAML parses and has the blocks the recipe reads, that the data paths exist, and that
every row of the train and val files is {"messages": [system, user, assistant]} with the
shared template as the system turn and a label the strict parser accepts.

--tokenizer also needs transformers and nemo-automodel (the GPU venv). It tokenizes every row
with the Qwen chat template and fails if any row is longer than seq_length, then builds the
real ChatDataset on row 0 and decodes the tokens that carry loss. Those must be the label
JSON plus "<|im_end|>", with no role header and no prompt text. That is the check that
the answer-only mask works as intended. Run it before training, it takes seconds.

Exit 0 = all checks passed, 1 = something failed (each failure is printed).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, load_template, parse_output, read_jsonl  # noqa: E402

DEFAULT_CONFIG = REPO / "train" / "lora_qwen25_1p5b.yaml"
REQUIRED_BLOCKS = ["recipe", "step_scheduler", "model", "peft", "optimizer", "loss_fn",
                   "dataset", "dataloader", "validation_dataset", "validation_dataloader", "checkpoint"]


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML is not installed: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        sys.exit(f"{path}: expected a mapping at the top level")
    return cfg


def resolve(path_str: str) -> Path:
    """Data paths in the YAML are relative to the repo root (where training is launched)."""
    p = Path(path_str)
    return p if p.is_absolute() else REPO / p


def check_config(cfg: dict) -> list:
    errors = []
    for block in REQUIRED_BLOCKS:
        if block not in cfg:
            errors.append(f"config is missing the '{block}' block")
    for block in ("dataset", "validation_dataset"):
        d = cfg.get(block) or {}
        if not isinstance(d.get("seq_length"), int):
            errors.append(f"{block}.seq_length must be an integer")
        if not d.get("path_or_dataset_id"):
            errors.append(f"{block}.path_or_dataset_id is not set")
        elif not resolve(d["path_or_dataset_id"]).is_file():
            errors.append(f"{block}.path_or_dataset_id: {d['path_or_dataset_id']} does not exist")
    return errors


def check_rows(path: Path, template: str) -> list:
    """Shape of each row in a NeMo chat file. Returns error strings (empty = fine)."""
    errors = []
    try:
        rows = read_jsonl(path)
    except (OSError, ValueError) as e:
        return [f"{path.name}: cannot read: {e}"]
    if not rows:
        return [f"{path.name}: no rows"]
    for i, r in enumerate(rows, 1):
        msgs = r.get("messages") if isinstance(r, dict) else None
        if not isinstance(msgs, list) or [m.get("role") for m in msgs] != ["system", "user", "assistant"]:
            errors.append(f"{path.name} row {i}: messages must be [system, user, assistant]")
        elif msgs[0].get("content") != template:
            errors.append(f"{path.name} row {i}: system turn differs from shared/template_v1.txt")
        elif not parse_output(msgs[2].get("content"))[1]:
            errors.append(f"{path.name} row {i}: assistant turn is not a valid label")
        if len(errors) >= 5:
            errors.append("(stopping after 5 errors)")
            break
    return errors


def check_tokens(cfg: dict) -> list:
    """GPU-venv checks: sequence lengths, and what the loss mask actually supervises."""
    try:
        from transformers import AutoTokenizer
        from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
    except ImportError as e:
        return [f"--tokenizer needs transformers and nemo-automodel installed: {e}"]

    errors = []
    tok = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_name_or_path"])
    for block in ("dataset", "validation_dataset"):
        d = cfg[block]
        rows = read_jsonl(resolve(d["path_or_dataset_id"]))
        lengths = [len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=True)["input_ids"])
                   for r in rows]
        print(f"  {block}: {len(rows)} rows, longest {max(lengths)} tokens, seq_length {d['seq_length']}")
        if max(lengths) > d["seq_length"]:
            errors.append(f"{block}: longest row ({max(lengths)}) is over seq_length ({d['seq_length']}); "
                          "it would be truncated")

    d = cfg["dataset"]
    kwargs = {k: d[k] for k in ("seq_length", "padding", "truncation", "mask_generation_prompt") if k in d}
    ds = ChatDataset(str(resolve(d["path_or_dataset_id"])), tok, **kwargs)
    ex = ds[0]
    supervised = [t for t in ex["labels"] if t != -100]
    text = tok.decode(supervised)
    # prep_nemo.py writes the label as compact JSON, so the expected span is that string + <|im_end|>
    want = read_jsonl(resolve(d["path_or_dataset_id"]))[0]["messages"][2]["content"] + "<|im_end|>"
    print(f"  row 0 supervised tokens ({len(supervised)} of {len(ex['labels'])}): {text!r}")
    if text.strip() != want:
        errors.append(f"loss mask: supervised text is {text!r}, expected {want!r}")
    return errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tokenizer", action="store_true", help="also check token lengths and the loss mask (GPU venv)")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    print(f"config: {args.config} parses")
    errors = check_config(cfg)
    if not errors:
        template = load_template()
        for block in ("dataset", "validation_dataset"):
            path = resolve(cfg[block]["path_or_dataset_id"])
            row_errors = check_rows(path, template)
            print(f"  {block}: {path.relative_to(REPO) if path.is_relative_to(REPO) else path}: "
                  f"{'ok' if not row_errors else 'FAILED'}")
            errors += row_errors
    if not errors and args.tokenizer:
        print("tokenizer checks:")
        errors += check_tokens(cfg)
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    print("preflight: " + ("FAILED" if errors else "ok"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
