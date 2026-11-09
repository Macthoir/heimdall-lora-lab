"""Checks on train/lora_qwen25_1p5b.yaml and scripts/preflight_train.py. No GPU, no NeMo install.

The YAML tests need PyYAML and are skipped without it (pip install pyyaml).
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import preflight_train  # noqa: E402
from common import REPO, load_template  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None

CONFIG = REPO / "train" / "lora_qwen25_1p5b.yaml"
REQUIREMENTS = REPO / "train" / "requirements-train.txt"
SERVE_REQUIREMENTS = REPO / "train" / "requirements-serve.txt"


@unittest.skipIf(yaml is None, "PyYAML not installed")
class TrainConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_has_every_block_the_recipe_reads(self):
        self.assertEqual(preflight_train.check_config(self.cfg), [])

    def test_fixed_hyperparameters(self):
        c = self.cfg
        self.assertEqual(c["seed"], 1)
        self.assertEqual(c["model"]["pretrained_model_name_or_path"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(c["model"]["torch_dtype"], "bf16")
        self.assertEqual(c["step_scheduler"]["num_epochs"], 3)
        self.assertEqual(c["optimizer"]["lr"], 1e-4)
        peft = c["peft"]
        self.assertEqual((peft["dim"], peft["alpha"], peft["dropout"]), (16, 32, 0.05))
        # All linear layers except lm_head: the 7 *_proj modules, not match_all_linear.
        self.assertEqual(peft["target_modules"], ["*_proj"])
        self.assertNotIn("match_all_linear", peft)
        self.assertEqual(c["distributed"]["tp_size"], 1)

    def test_loss_on_assistant_turn_only(self):
        for block in ("dataset", "validation_dataset"):
            self.assertTrue(self.cfg[block]["mask_generation_prompt"], block)
            self.assertNotIn("answer_only_loss_mask", self.cfg[block])  # default True; must not be turned off

    def test_val_each_epoch_and_final_checkpoint_kept(self):
        s = self.cfg["step_scheduler"]
        self.assertTrue(s["save_checkpoint_every_epoch"])
        self.assertTrue(s["validate_on_checkpoint"])
        self.assertGreater(s["ckpt_every_steps"], 1000)  # no mid-epoch saves
        self.assertEqual(self.cfg["checkpoint"]["checkpoint_dir"].rstrip("/"), "checkpoints/lora-v1")
        self.assertEqual(self.cfg["checkpoint"]["model_save_format"], "safetensors")
        self.assertIn("lr_scheduler", CONFIG.read_text(encoding="utf-8"))  # only in a comment
        self.assertNotIn("lr_scheduler", self.cfg)

    def test_data_files_are_the_nemo_chat_files(self):
        self.assertEqual(self.cfg["dataset"]["path_or_dataset_id"], "data/nemo/train.jsonl")
        self.assertEqual(self.cfg["validation_dataset"]["path_or_dataset_id"], "data/nemo/val.jsonl")

    def test_seq_length_covers_the_measured_longest_row(self):
        # 1305 tokens measured with the Qwen2.5 tokenizer on the current data/nemo/train.jsonl.
        for block in ("dataset", "validation_dataset"):
            self.assertGreaterEqual(self.cfg[block]["seq_length"], 1305)

    def test_pinned_version_matches_requirements(self):
        pin = re.search(r"^nemo-automodel==(\S+)", REQUIREMENTS.read_text(encoding="utf-8"), re.M)
        self.assertIsNotNone(pin)
        self.assertIn(f"nemo-automodel=={pin.group(1)}", CONFIG.read_text(encoding="utf-8"))

    def test_vllm_and_nemo_are_in_separate_venvs(self):
        # vllm 0.32.0 and nemo-automodel 0.7.0 pin different quack-kernels; one venv is unsatisfiable.
        train = REQUIREMENTS.read_text(encoding="utf-8")
        serve = SERVE_REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^vllm", train, re.M))
        self.assertIsNone(re.search(r"^nemo-automodel", serve, re.M))
        self.assertIsNotNone(re.search(r"^vllm==", serve, re.M))


class NemoChatFiles(unittest.TestCase):
    """Our JSONL is what the preflight (and ChatDataset's local-file reader) expects."""

    def test_real_files_pass(self):
        template = load_template()
        for name in ("train", "val"):
            self.assertEqual(preflight_train.check_rows(REPO / "data" / "nemo" / f"{name}.jsonl", template), [], name)

    def test_bad_rows_are_caught(self):
        template = load_template()
        good = {"messages": [{"role": "system", "content": template}, {"role": "user", "content": "hi"},
                             {"role": "assistant", "content": '{"task_type":"debugging","complexity":"low"}'}]}
        wrong_system = json.loads(json.dumps(good))
        wrong_system["messages"][0]["content"] = "something else"
        bad_label = json.loads(json.dumps(good))
        bad_label["messages"][2]["content"] = "debugging, low"
        no_assistant = {"messages": good["messages"][:2]}
        with tempfile.TemporaryDirectory() as d:
            for name, row, expect in (("good", good, None), ("sys", wrong_system, "system turn differs"),
                                      ("label", bad_label, "not a valid label"),
                                      ("short", no_assistant, "must be [system, user, assistant]")):
                p = Path(d) / f"{name}.jsonl"
                p.write_text(json.dumps(row) + "\n", encoding="utf-8")
                errors = preflight_train.check_rows(p, template)
                if expect is None:
                    self.assertEqual(errors, [])
                else:
                    self.assertTrue(any(expect in e for e in errors), (name, errors))


if __name__ == "__main__":
    unittest.main()
