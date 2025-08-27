from ocr import OCRRunner
from preprocessing import Labeler
from tokenizer import Tokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification
from torch.optim import AdamW

ocr_runner = OCRRunner()
labeler = Labeler()
tokenizer = Tokenizer()

#ocr_runner.pipeline()
#labeler.process_json_files()
all_encodings = tokenizer.tokenize_labeled()
# print(all_encodings)

dataset = TensorDataset(
    all_encodings["input_ids"],
    all_encodings["attention_mask"],
    all_encodings["bbox"],
    all_encodings["labels"]
)

train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
#print(train_loader)

for k, batch in enumerate(train_loader):
    print(f"step: {k}, batch: {batch}")
    input_ids, attention_mask, bbox, labels = [b for b in batch]
    print(input_ids)
    print(attention_mask)
    print(bbox)
    print(labels)
    break

# num_labels = 6  # change according to your dataset

# model = LayoutLMv3ForTokenClassification.from_pretrained(
#     "microsoft/layoutlmv3-base",
#     num_labels=num_labels
# )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# optimizer = AdamW(model.parameters(), lr=5e-5)

# epochs = 2  # just for demonstration

# for epoch in range(epochs):
#     print(f"\nEpoch {epoch+1}/{epochs}")
#     model.train()  # set model to training mode
    
#     for step, batch in enumerate(train_loader):
#         input_ids, attention_mask, bbox, labels = [b.to(device) for b in batch]
