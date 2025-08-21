import json 
from PIL import Image

words = []
boxes = []

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

labels_map = {
    "O": 0,                 
    "NUMERO_NF": 1,
    "DATA_EMISSAO": 2,
    "DATA_COMPETENCIA": 3,
    "NOME_CREDOR": 4,
    "VALOR_NF": 5,
}

labeled_tokens = []

with open("./output/invoice_page_1_res.json", "r", encoding="utf-8") as f:
    ocr_data = json.load(f)
    img = Image.open("invoice_page_1.png")
    width, height = img.size

    for i, (text, box) in enumerate(zip(ocr_data["rec_texts"], ocr_data["rec_polys"])):
        print(f"{i}: {text}")
        label = input(f"Enter label for token '{text}' (e.g. INVOICE_NUMBER, O): ").strip().upper()
        if label not in labels_map:
            label = "O"

        normalized_bbox = normalize_box(box, width, height)
        labeled_tokens.append({"text": text, "bbox": normalized_bbox, "label": labels_map[label]})

with open("labeled_invoice.json", "w", encoding="utf-8") as f:
    json.dump(labeled_tokens, f, ensure_ascii=False, indent=4)


#         words.append(text)
#         boxes.append(normalize_box(box, width, height))

# print("First 5 words:", words[:5])
# print("First 5 boxes:", boxes[:5])
