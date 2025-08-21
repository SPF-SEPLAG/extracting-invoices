from pdf2image import convert_from_path

pdf_path = "./invoices/5023.pdf"
pages = convert_from_path(pdf_path)
page = pages[0]
page.save("invoice_page_1.png")

