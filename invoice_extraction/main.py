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

## Debugging all_encodings
#print(all_encodings)
# print("\n--- Encodings summary ---")
# for k, v in all_encodings.items():
#     print(f"{k:15} shape={v.shape} dtype={v.dtype}")
#     # Peek at the first row only (to confirm it looks sane)
#     print(f"  first row (truncated): {v[0][:10]}\n")


# dataset = TensorDataset(
#     all_encodings["input_ids"],
#     all_encodings["attention_mask"],
#     all_encodings["bbox"],
#     all_encodings["labels"]
# )

# train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
# #print(train_loader)

# model = LayoutLMv3ForTokenClassification.from_pretrained(
#     "microsoft/layoutlmv3-base",
#     num_labels=6
# )
# print(model.config.max_position_embeddings)  # should print 512

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# optimizer = AdamW(model.parameters(), lr=5e-5)

# for k, batch in enumerate(train_loader):
#     #print(f"step: {k}, batch: {batch}")
#     input_ids, attention_mask, bbox, labels = [b.to(device) for b in batch]

#     # Forward pass
#     outputs = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         bbox=bbox,
#         labels=labels
#     )
    
#     loss = outputs.loss
#     logits = outputs.logits  # shape: [batch_size, seq_len, num_labels]
    
#     # Backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if k % 2 == 0:  # print every 2 steps
#         print(f"  Step {k}, Loss: {loss.item():.4f}")


# optimizer = AdamW(model.parameters(), lr=5e-5)

# epochs = 2  # just for demonstration

# for epoch in range(epochs):
#     print(f"\nEpoch {epoch+1}/{epochs}")
#     model.train()  # set model to training mode
    
#     for step, batch in enumerate(train_loader):
#         input_ids, attention_mask, bbox, labels = [b.to(device) for b in batch]
