from .bbox_utils import normalize_box
from pathlib import Path
import json
from PIL import Image

class Labeler:
    def __init__(self, supplier="prodemge", base_dir="./../data"):
        self.supplier = supplier
        self.base_dir = Path(base_dir)

        self.output_img_folder = self.base_dir/"output"/supplier/"png"
        self.output_json_folder = self.base_dir/"output"/supplier/"json"
        self.output_labeled_folder = self.base_dir/"output"/supplier/"labeled"

        # Create if doesnt exist
        self.output_labeled_folder.mkdir(parents=True, exist_ok=True)

        self.labels_map = {
            "O": 0,                 
            "NUMERO_NF": 1,
            "DATA_EMISSAO": 2,
            "DATA_COMPETENCIA": 3,
            "NOME_CREDOR": 4,
            "VALOR_NF": 5,
        }

    def process_json_files(self):
        for json_file in self.output_json_folder.glob("*.json"):
            labeled_tokens = []
            with open(json_file, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)
                img = Image.open(f"{self.output_img_folder}/{json_file.stem.replace("_res", "")}.png")
                width, height = img.size

                for i, (text, box) in enumerate(zip(ocr_data["rec_texts"], ocr_data["rec_polys"])):
                    print(f"{i}: {text}")
                    label = input(f"Enter label for token '{text}' (e.g. INVOICE_NUMBER, O): ").strip().upper()
                    if label not in self.labels_map:
                        label = "O"

                    normalized_bbox = normalize_box(box, width, height)
                    labeled_tokens.append({"text": text, "bbox": normalized_bbox, "label": self.labels_map[label]})

            with open(f"{self.output_labeled_folder}/labeled_{json_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(labeled_tokens, f, ensure_ascii=False, indent=4)


