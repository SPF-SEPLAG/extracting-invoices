import json
from PIL import Image

# 1. Load your OCR output JSON file
with open("ocr_output.json", "r", encoding="utf-8") as f:
    ocr_data = json.load(f)

# 2. Load your invoice image to get dimensions
image = Image.open("invoice_page_1.png")
width, height = image.size

# 3. Function to normalize bounding boxes (4 points -> xmin, ymin, xmax, ymax)
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

# 4. Define label mapping
labels_map = {
        "O": 0,                  # Outside any field
            "INVOICE_NUMBER": 1,
                "INVOICE_DATE": 2,
                    "TOTAL_AMOUNT": 3,
                        "SUPPLIER_NAME": 4,
                            # Add more fields as needed
                            }
                            

# 5. Prepare list for labeled tokens
labeled_tokens = []

# 6. Example of manual labeling (modify as needed)
# You can automate this or make a loop to input labels for each token

print("Starting manual labeling...")
for i, (text, box) in enumerate(zip(ocr_data['res']['rec_texts'], ocr_data['res']['rec_polys'])):
    print(f"{i}: {text}")
    label = input(f"Enter label for token '{text}' (e.g. INVOICE_NUMBER, O): ").strip().upper()
    if label not in labels_map:
        print(f"Label '{label}' not recognized. Defaulting to 'O'.")
        label = "O"

    normalized_bbox = normalize_box(box, width, height)

    labeled_tokens.append({
        "text": text,
        "bbox": normalized_bbox,
        "label": labels_map[label]
    })

# 7. Save labeled tokens to JSON for training
with open("labeled_invoice.json", "w", encoding="utf-8") as f:
    json.dump(labeled_tokens, f, ensure_ascii=False, indent=4)

print("Labeled data saved to 'labeled_invoice.json'")
