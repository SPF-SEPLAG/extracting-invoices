from pathlib import Path
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

class OCRRunner:
    def __init__(self, supplier = "prodemge", base_dir = "./data"):
        self.supplier = supplier
        self.base_dir = Path(base_dir)

        self.pdf_folder = self.base_dir/"invoices"/supplier
        self.output_img_folder = self.base_dir/"output"/supplier/"png"
        self.output_json_folder = self.base_dir/"output"/supplier/"json"

        # Create if doesnt exist
        self.output_img_folder.mkdir(parents=True, exist_ok=True)
        self.output_json_folder.mkdir(parents=True, exist_ok=True)

        self.ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    def pdfs_to_imgs(self):
        for pdf_file in self.pdf_folder.glob("*.pdf"):
            pages = convert_from_path(pdf_file)
            for i, page in enumerate(pages):
                page.save(f"{self.output_img_folder}/{pdf_file.stem}_page_{i+1}.png")
        
    def img_to_json(self):
        for png_file in self.output_img_folder.glob("*.png"):
            result = self.ocr.predict(str(png_file))
            for res in result:
                res.save_to_json(self.output_json_folder)

    def pipeline(self):
        self.pdfs_to_imgs()
        self.img_to_json()





