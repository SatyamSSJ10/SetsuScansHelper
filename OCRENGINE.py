import jaconv
import re
from PIL import Image
class OCREngine:
    """
    A simple class that instantiates the selected OCR engine and 
    provides a uniform 'predict' method.
    """
    def __init__(self, engine_name):
        self.engine_name = engine_name
        self.reader = None
        self.pretrained_model_name_or_path = None
        self.feature_extractor = None
        self.model = None
        if engine_name == "Chinese":
            print("Using Chinese")
            import easyocr
            self.reader = easyocr.Reader(['ch_sim'])

        elif engine_name == "Japanese":
            from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel

            self.pretrained_model_name_or_path = r"./SSHelper"
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.pretrained_model_name_or_path)

            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.pretrained_model_name_or_path)

    def post_process(self, text):
        if self.engine_name == "Chinese":
            return ((((text.replace('[','')).replace('\'','')).replace(']','')).replace(':.','...')).trplsvr(':','...')
        text = ''.join(text.split())
        text = text.replace('…', '...')
        text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
        text = jaconv.h2z(text, ascii=True, digit=True)
        return text
    
    def preprocess(self, img):
        img = Image.fromarray(img)
        img = img.convert('L').convert('RGB')
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()
    
    def predict(self, np_image):
        """
        Takes a NumPy array of the image/ROI and returns recognized text lines.
        """
        if self.engine_name == "Chinese" and self.reader:
            # Example usage of EasyOCR
            return self.reader.readtext(
                np_image, detail=0, paragraph=True, y_ths=1, canvas_size=1000
            )
        elif self.engine_name == "Japanese" and self.feature_extractor and self.tokenizer and self.model:
            x = self.preprocess(np_image)
            x = self.model.generate(x[None].to(self.model.device), max_length=300)[0] #.cpu()
            x = self.tokenizer.decode(x, skip_special_tokens=True)
            x = self.post_process(x)
            return [x]
        
    def cleanup(self):
        if self.reader or self.model:
            import torch
            torch.cuda.empty_cache()
            pass
        self.reader = None
        self.model = None
