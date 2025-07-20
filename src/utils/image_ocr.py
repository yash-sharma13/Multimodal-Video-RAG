import logging
import re
from PIL import Image, ImageEnhance
import pytesseract
from spellchecker import SpellChecker

# Set up logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def enhance_image_for_ocr(image_path: str) -> Image.Image:
    try:
        img = Image.open(image_path).convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        return img
    except Exception as e:
        logging.warning(f"Error enhancing image {image_path}: {str(e)}")
        return Image.open(image_path)

def extract_ocr_text(image_path: str) -> tuple:
    try:
        img = enhance_image_for_ocr(image_path)
        text = pytesseract.image_to_string(img, lang='eng')
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        spell_checker = SpellChecker()
        words = cleaned_text.split()
        ocr_quality = 0.0
        if words:
            unknown = sum(1 for word in words if not spell_checker.known([word.lower()]))
            coherence = 1 - (unknown / len(words))
            ocr_quality = min(len(words) / 50, 1.0) * coherence
            if len(words) < 5:
                ocr_quality = min(0.1, ocr_quality)
        logging.debug(f"Extracted OCR text from {image_path}: {cleaned_text[:50]}..., quality={ocr_quality:.3f}")
        return cleaned_text, ocr_quality
    except Exception as e:
        logging.warning(f"Error extracting OCR text: {str(e)}")
        return "", 0.0 