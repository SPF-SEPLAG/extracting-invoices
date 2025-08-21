from pathlib import Path
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

class OCRRunner:
    def __init__(self, supplier, base_dir = "./../data"):
        self.supplier = supplier
        self.base_dir = Path(base_dir)

        self.pdf_folder = self.base_dir/"invoices"/supplier


        self.ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False) 

    def pdfs_to_imgs(self):
        pdf_folder = Path("./../data/invoices/prodemge")
        output_folder = Path("./../data/output/prodemge/png")
        output_folder.mkdir(parents=True, exist_ok=True)

        for pdf_file in pdf_folder.glob("*.pdf"):
            pages = convert_from_path(pdf_file)
            for i, page in enumerate(pages):
                page.save(f"{output_folder}/{pdf_file.stem}_page_{i+1}.png")
        
    def img_to_json(self):
        output_folder = Path("./../data/output/prodemge/png")
        for png_file in output_folder.glob("*.png"):
            result = self.ocr.predict(str(png_file))
            for res in result:
                res.save_to_json("./../data/output/prodemge/json")




