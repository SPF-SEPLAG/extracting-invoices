from transformers import pipeline


nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

print(nlp(
    "invoice_page_1.png",
    "What is the CNPJ from PRODEMGE?"
))
