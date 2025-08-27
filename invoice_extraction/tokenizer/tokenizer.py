from pathlib import Path
import json
from transformers import LayoutLMv3TokenizerFast
import torch

class Tokenizer:
    def __init__(self, supplier = "prodemge", base_dir = "./../data"):
        self.supplier = supplier
        self.base_dir = Path(base_dir)
        self.output_labeled_folder = self.base_dir/"output"/supplier/"labeled"
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

    def tokenize_labeled(self):
        all_encodings = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "bbox": [],
        }

        for labeled_file in self.output_labeled_folder.glob("labeled_*.json"):
            with open(labeled_file, "r", encoding="utf-8") as f:
                labeled_data = json.load(f)
                words = [item["text"] for item in labeled_data]
                boxes = [item["bbox"] for item in labeled_data]
                labels = [item["label"] for item in labeled_data]

                encoding = self.tokenizer(words, boxes=boxes, word_labels=labels, padding="max_length", truncation=True, return_tensors="pt", max_length=1100)
                # Appends encoding to dict of lists all_encodings
                all_encodings["input_ids"].append(encoding["input_ids"][0])
                all_encodings["attention_mask"].append(encoding["attention_mask"][0])
                all_encodings["labels"].append(encoding["labels"][0])
                all_encodings["bbox"].append(encoding["bbox"][0])

        # Convert encoding to tensor
        for k, v in all_encodings.items():
            v = torch.stack(v)
            all_encodings[k] = v

        return all_encodings
