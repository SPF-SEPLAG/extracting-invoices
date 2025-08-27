from ocr import OCRRunner
from preprocessing import Labeler
from tokenizer import Tokenizer

ocr_runner = OCRRunner()
labeler = Labeler()
tokenizer = Tokenizer()

#ocr_runner.pipeline()
#labeler.process_json_files()
print(tokenizer.tokenize_labeled())
