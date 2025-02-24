from multiprocessing import Pool
import os
import cv2
import numpy as np
import json
from paddleocr import PaddleOCR
from pathlib import Path
from utils import (
    enhance_image,
    resize_image_if_needed,
    create_annotated_image,
    FIELD_LABELS,
    COUNTRY_CODES,
    EXCLUDED_WORDS,
    SUPPORTED_LANGUAGES,
    logger
)

class HealthCardInfo:
    def __init__(self):
        self.insurance_number = ""
        self.surname = ""
        self.first_name = ""
        self.birth_date = ""
        self.personal_number = ""
        self.insurance_code = ""  # 4-digit Swiss insurance code
        self.insurance_name = ""  # Insurance provider name
        self.card_number = ""
        self.expiry_date = ""
        self.detected_language = ""

    def to_dict(self):
        return {
            "insurance_number": self.insurance_number,
            "surname": self.surname,
            "first_name": self.first_name,
            "birth_date": self.birth_date,
            "personal_number": self.personal_number,
            "insurance_code": self.insurance_code,
            "insurance_name": self.insurance_name,
            "card_number": self.card_number,
            "expiry_date": self.expiry_date,
            "detected_language": self.detected_language
        }

def detect_card_language(results):
    """Detect the language of the card based on field labels."""
    language_scores = {lang: 0 for lang in SUPPORTED_LANGUAGES}
    
    for result in results:
        text = result[1][0]  # PaddleOCR format: [[[points]], [text, confidence]]
        prob = result[1][1]
        
        if prob < 0.4:
            continue
            
        text = text.strip()
        for field in FIELD_LABELS:
            for lang in SUPPORTED_LANGUAGES:
                if any(label in text for label in FIELD_LABELS[field][lang]):
                    language_scores[lang] += 1
    
    # Get the language with the highest score
    detected_lang = max(language_scores.items(), key=lambda x: x[1])[0]
    logger.debug(f"Detected language: {detected_lang} (scores: {language_scores})")
    return detected_lang

def extract_card_info(results):
    card_info = HealthCardInfo()
    
    # First detect the language
    detected_lang = detect_card_language(results)
    card_info.detected_language = detected_lang
    
    print(f"\nDetected language: {detected_lang}")
    print("\nDetected text:")
    for result in results:
        text = result[1][0]
        prob = result[1][1]
        print(f"{text} ({prob:.2%})")
    
    detected_values = {}
    potential_names = []
    
    for idx, result in enumerate(results):
        bbox = result[0]
        text = result[1][0]
        prob = result[1][1]
        
        text = text.strip()
        
        if prob < 0.4:
            continue

        if text in COUNTRY_CODES:
            continue
            
        # Improved name detection - must be longer than 2 characters and not contain numbers
        if text.isupper() and len(text) > 2 and prob > 0.7 and text.isalpha():
            if not any(text in label for labels in FIELD_LABELS.values() for label in labels[detected_lang]) and \
               not any(word in text for word in EXCLUDED_WORDS[detected_lang]):
                x_min = min(point[0] for point in bbox)
                y_min = min(point[1] for point in bbox)
                potential_names.append((text, y_min, x_min))
        
        # Check field labels in detected language
        if any(label in text for label in FIELD_LABELS['insurance_number'][detected_lang]):
            number = ''.join(filter(str.isdigit, text))
            if number and len(number) >= 6:
                detected_values['insurance_number'] = number
            elif idx + 1 < len(results):
                next_text = results[idx + 1][1][0].strip()
                number = ''.join(filter(str.isdigit, next_text))
                if number and len(number) >= 6:
                    detected_values['insurance_number'] = number
        
        if "756" in text and "." in text and prob > 0.4:
            detected_values['personal_number'] = text.strip()
        
        # Simplified insurance detection
        if '-' in text and prob > 0.5:  # Format like "0032-Aquilana" or "0032 - Helsana"
            parts = [p.strip() for p in text.split('-')]
            if len(parts) == 2:
                code = ''.join(filter(str.isdigit, parts[0]))
                if len(code) == 4 or len(code) == 5:  # Allow both 4 and 5 digit codes
                    detected_values['insurance_code'] = code
                    # Only take the first word of the second part as insurance name
                    detected_values['insurance_name'] = parts[1].split()[0]
        
        if len(text) == 10 and text.count("/") == 2:
            try:
                day, month, year = text.split("/")
                if day.isdigit() and month.isdigit() and year.isdigit():
                    if len(year) == 4 and 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
                        if 'birth_date' not in detected_values:
                            detected_values['birth_date'] = text
                        else:
                            detected_values['expiry_date'] = text
            except:
                pass
        
        if text.startswith("80756") and len(text) > 15 and prob > 0.7:
            detected_values['card_number'] = text
    
    potential_names.sort(key=lambda x: (x[1], x[2]))  # Sort by y position first, then x position
    
    # Improved name assignment
    for result in results:
        text = result[1][0].strip()
        prob = result[1][1]
        
        # Look for name field labels
        if any(label in text for label in FIELD_LABELS['surname'][detected_lang]):
            # Find the closest name after this label
            label_y = min(point[1] for point in result[0])
            closest_name = None
            min_distance = float('inf')
            
            for name, y, _ in potential_names:
                distance = abs(y - label_y)
                if distance < min_distance and y >= label_y:
                    min_distance = distance
                    closest_name = name
            
            if closest_name and min_distance < 100:  # Only assign if within reasonable distance
                detected_values['surname'] = closest_name
                potential_names = [n for n in potential_names if n[0] != closest_name]
        
        elif any(label in text for label in FIELD_LABELS['first_name'][detected_lang]):
            # Find the closest name after this label
            label_y = min(point[1] for point in result[0])
            closest_name = None
            min_distance = float('inf')
            
            for name, y, _ in potential_names:
                distance = abs(y - label_y)
                if distance < min_distance and y >= label_y:
                    min_distance = distance
                    closest_name = name
            
            if closest_name and min_distance < 100:  # Only assign if within reasonable distance
                detected_values['first_name'] = closest_name
                potential_names = [n for n in potential_names if n[0] != closest_name]
    
    # If names weren't assigned by labels, use the remaining names in order
    if potential_names and 'surname' not in detected_values:
        detected_values['surname'] = potential_names[0][0]
        if len(potential_names) > 1:
            detected_values['first_name'] = potential_names[1][0]
    
    # Update card_info with detected values
    for field, value in detected_values.items():
        setattr(card_info, field, value)
    
    return card_info

def process_image_ocr(image):
    """
    Process an image through OCR and return the results.
    Args:
        image: numpy array of the image
    Returns:
        results: list of OCR results
    """
    enhanced = enhance_image(image)
    ocr = PaddleOCR(use_angle_cls=True, lang='german', show_log=False)
    results = ocr.ocr(enhanced, cls=True)
    return results[0] if results else []

def process_single_image(image_path):
    try:
        output_dir = "detected_results"
        json_dir = "card_data"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        image = resize_image_if_needed(image)
        
        results = process_image_ocr(image)
        
        if not results:
            print(f"No text detected in {image_path}")
            return
        
        card_info = extract_card_info(results)
        
        json_filename = f"card_data_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        json_path = os.path.join(json_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(card_info.to_dict(), f, ensure_ascii=False, indent=2)
        
        annotated = create_annotated_image(image, results)
        output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated)
        
        print(f"\nExtracted Card Information for {os.path.basename(image_path)}:")
        for key, value in card_info.to_dict().items():
            if value:
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_images(directory_path):
    output_dir = "detected_results"
    json_dir = "card_data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with Pool() as pool:
        pool.map(process_single_image, image_files)

if __name__ == "__main__":
    desktop_path = str(Path.home() / "Desktop" / "ids")
    if not os.path.exists(desktop_path):
        print("Please create an 'ids' folder on your Desktop and place the card images there.")
    else:
        process_images(desktop_path)
