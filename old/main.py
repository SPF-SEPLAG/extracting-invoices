from pdf2image import convert_from_path
from pathlib import Path
from paddleocr import PaddleOCR
import json 
from PIL import Image
from transformers import LayoutLMv3TokenizerFast
import torch

def pdfs_to_imgs():
    pdf_folder = Path("./invoices/prodemge")
    output_folder = Path("./output/prodemge/png")
    output_folder.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_folder.glob("*.pdf"):
        pages = convert_from_path(pdf_file) 
        for i, page in enumerate(pages):
            page.save(f"{output_folder}/{pdf_file.stem}_page_{i+1}.png")

def img_to_json(ocr: PaddleOCR):
    output_folder = Path("./output/prodemge/png")
    for png_file in output_folder.glob("*.png"):
        result = ocr.predict(str(png_file))
        for res in result:
            res.save_to_json("./output/prodemge/json")

# Function to normalize bounding boxes (4 points -> xmin, ymin, xmax, ymax)
def normalize_box(box, width, height):
    # box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [
        int(1000 * (xmin / width)),
        int(1000 * (ymin / height)),
        int(1000 * (xmax / width)),
        int(1000 * (ymax / height)),
    ]

def process_json():
    labels_map = {
        "O": 0,                 
        "NUMERO_NF": 1,
        "DATA_EMISSAO": 2,
        "DATA_COMPETENCIA": 3,
        "NOME_CREDOR": 4,
        "VALOR_NF": 5,
    }

    output_folder = Path("./output/prodemge/json")
    for json_file in output_folder.glob("*.json"):
        labeled_tokens = []
        with open(json_file, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
            img = Image.open(f"./output/prodemge/png/{json_file.stem.replace("_res", "")}.png")
            width, height = img.size

            for i, (text, box) in enumerate(zip(ocr_data["rec_texts"], ocr_data["rec_polys"])):
                print(f"{i}: {text}")
                label = input(f"Enter label for token '{text}' (e.g. INVOICE_NUMBER, O): ").strip().upper()
                if label not in labels_map:
                    label = "O"

                normalized_bbox = normalize_box(box, width, height)
                labeled_tokens.append({"text": text, "bbox": normalized_bbox, "label": labels_map[label]})

        with open(f"./output/prodemge/labeled/labeled_{json_file.stem}.json", "w", encoding="utf-8") as f:
            json.dump(labeled_tokens, f, ensure_ascii=False, indent=4)

def tokenize_labeled(tokenizer):
    all_encodings = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "bbox": [],
    }

    labeled_folder = Path("./output/prodemge/labeled")
    for labeled_file in labeled_folder.glob("labeled_*.json"):
        with open(labeled_file, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)
            words = [item["text"] for item in labeled_data]
            boxes = [item["bbox"] for item in labeled_data]
            labels = [item["label"] for item in labeled_data]

            encoding = tokenizer(words, boxes=boxes, word_labels=labels, padding="max_length", truncation=True, return_tensors="pt", max_length=1100)
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






#            print(encoding.keys())
            
            # token_ids = encoding["input_ids"][0]
            # token_labels = encoding["labels"][0]
            # token_boxes = encoding["bbox"][0]
            # attention_masks = encoding["attention_mask"][0]

            # tokens = tokenizer.convert_ids_to_tokens(token_ids)
#            print(encoding["input_ids"].shape)

            # for token_id, token, box, label, attention_mask in zip(token_ids, tokens, token_boxes, token_labels, attention_masks):
            #     print(f"ID: {token_id.item():6} | Token: {token:15} | BBox: {box.tolist()} | Label: {label.item()} | Attention mask: {attention_mask}")



# pdfs_to_imgs()

# ocr = PaddleOCR(
#     use_doc_orientation_classify=False, 
#     use_doc_unwarping=False, 
#     use_textline_orientation=False) 

# img_to_json(ocr)

# process_json()

tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
tokenize_labeled(tokenizer)