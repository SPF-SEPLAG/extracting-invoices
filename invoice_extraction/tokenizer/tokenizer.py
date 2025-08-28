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
        chunk_size = 514
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

                encoding = self.tokenizer(words, boxes=boxes, word_labels=labels, padding="max_length", truncation=True, return_tensors="pt", max_length=1542)

                # ## Debugging
                # token_ids = encoding["input_ids"][0]
                # token_labels = encoding["labels"][0]
                # token_boxes = encoding["bbox"][0]
                # tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
                # for token_id, token, box, label in zip(token_ids, tokens, token_boxes, token_labels):
                #     print(f"ID: {token_id.item():6} | Token: {token:15} | BBox: {box.tolist()} | Label: {label.item()}")
                # print(encoding["input_ids"].shape)
                # print(encoding["labels"].shape)
                # print(encoding["bbox"].shape)

                # # Appends encoding to dict of lists all_encodings
                # all_encodings["input_ids"].append(encoding["input_ids"][0])
                # all_encodings["attention_mask"].append(encoding["attention_mask"][0])
                # all_encodings["labels"].append(encoding["labels"][0])
                # all_encodings["bbox"].append(encoding["bbox"][0])

                # Get token-level tensors
                input_ids = encoding["input_ids"][0]
                attention_mask = encoding["attention_mask"][0]
                token_labels = encoding["labels"][0]
                token_boxes = encoding["bbox"][0]

                # Chunk into pieces of chunk_size
                for i in range(0, input_ids.size(0), chunk_size):
                    all_encodings["input_ids"].append(input_ids[i:i+chunk_size])
                    all_encodings["attention_mask"].append(attention_mask[i:i+chunk_size])
                    all_encodings["labels"].append(token_labels[i:i+chunk_size])
                    all_encodings["bbox"].append(token_boxes[i:i+chunk_size])

        # Convert encoding to tensor
        for k, v in all_encodings.items():
            v = torch.stack(v)
            all_encodings[k] = v

        return all_encodings
