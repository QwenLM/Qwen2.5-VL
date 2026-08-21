import ast
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


FINETUNE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(FINETUNE_ROOT))
data_processor = importlib.import_module("qwenvl.data.data_processor")


class DataModuleTokenizerTests(unittest.TestCase):
    def test_collator_uses_training_tokenizer_max_length(self):
        processor_tokenizer = SimpleNamespace(pad_token_id=0, model_max_length=32)
        training_tokenizer = SimpleNamespace(pad_token_id=0, model_max_length=3)
        processor = SimpleNamespace(tokenizer=processor_tokenizer)
        data_args = SimpleNamespace(data_flatten=False, data_packing=False)

        with mock.patch.object(data_processor, "LazySupervisedDataset", return_value=object()):
            data_module = data_processor.make_supervised_data_module(
                processor,
                data_args,
                tokenizer=training_tokenizer,
            )

        instance = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "labels": torch.tensor([[1, 2, 3, 4, 5]]),
            "position_ids": torch.arange(5).view(1, 1, 5).expand(3, 1, 5),
        }
        batch = data_module["data_collator"]([instance])

        self.assertIs(data_module["data_collator"].tokenizer, training_tokenizer)
        self.assertEqual(batch["input_ids"].shape, (1, 3))
        self.assertEqual(batch["labels"].shape, (1, 3))
        self.assertEqual(batch["position_ids"].shape, (3, 1, 3))

    def test_training_entrypoint_forwards_configured_tokenizer(self):
        train_path = FINETUNE_ROOT / "qwenvl" / "train" / "train_qwen.py"
        tree = ast.parse(train_path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "make_supervised_data_module"
        ]

        self.assertEqual(len(calls), 1)
        tokenizer_keyword = next(
            keyword for keyword in calls[0].keywords if keyword.arg == "tokenizer"
        )
        self.assertIsInstance(tokenizer_keyword.value, ast.Name)
        self.assertEqual(tokenizer_keyword.value.id, "tokenizer")


if __name__ == "__main__":
    unittest.main()
