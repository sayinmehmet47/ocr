from multiprocessing import Pool
import os
import cv2
import numpy as np
import json
import easyocr
from pathlib import Path

class HealthCardInfo:
    def __init__(self):
        self.insurance_number = ""
        self.surname = ""
        self.first_name = ""
        self.birth_date = ""
        self.personal_number = ""
        self.insurance_provider_id = ""
        self.card_number = ""
        self.expiry_date = ""

    def to_dict(self):
        return {
            "insurance_number": self.insurance_number,
            "surname": self.surname,
            "first_name": self.first_name,
            "birth_date": self.birth_date,
            "personal_number": self.personal_number,
            "insurance_provider_id": self.insurance_provider_id,
            "card_number": self.card_number,
            "expiry_date": self.expiry_date
        }

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=10)
    return enhanced

def extract_card_info(results):
    FIELD_LABELS = {
        'insurance_number': ['Versicherten-Nr', 'Versicherten-Nr.', 'ersicherten-Nr', 'ersicherten-Nr.'],
        'surname': ['3. Name', 'Name', '3.Name'],
        'first_name': ['4. Vornamen', 'Vornamen', '4.Vornamen'],
        'birth_date': ['5. Geburtsdatum', 'Geburtsdatum', '5.Geburtsdatum', 'Geburtsdat'],
        'personal_number': ['6. Persönliche Kennnummer', 'Kennnummer', '6. Personliche'],
        'insurance_provider_id': ['7. Kennnummer des Trägers', '7. Kennnummer'],
        'card_number': ['8. Kennnummer der Karte'],
        'expiry_date': ['9. Ablaufdatum']
    }

    COUNTRY_CODES = ['CH']

    card_info = HealthCardInfo()
    
    print("\nDetected text:")
    for _, text, prob in results:
        print(f"{text} ({prob:.2%})")
    
    detected_values = {}
    potential_names = []
    
    for idx, (bbox, text, prob) in enumerate(results):
        text = text.strip()
        
        if prob < 0.4:
            continue

        if text in COUNTRY_CODES:
            continue
            
        if text.isupper() and len(text) > 1 and prob > 0.7:
            if not any(text in label for labels in FIELD_LABELS.values() for label in labels) and \
               not any(word in text for word in ["KARTE", "VERSICHERUNG", "EUROPEAN", "EUROPÄISCHE"]):
                x_min = min(point[0] for point in bbox)
                y_min = min(point[1] for point in bbox)
                potential_names.append((text, y_min, x_min))
        
        if any(label in text for label in FIELD_LABELS['insurance_number']):
            number = ''.join(filter(str.isdigit, text))
            if number and len(number) >= 6:
                detected_values['insurance_number'] = number
            elif idx + 1 < len(results):
                next_text = results[idx + 1][1].strip()
                number = ''.join(filter(str.isdigit, next_text))
                if number and len(number) >= 6:
                    detected_values['insurance_number'] = number
        
        if "756" in text and "." in text and prob > 0.4:
            detected_values['personal_number'] = text.strip()
        
        if prob > 0.6:
            if text == "0032":
                provider_id = text
                if idx + 1 < len(results) and "Aquilana" in results[idx + 1][1]:
                    provider_name = results[idx + 1][1].strip()
                    detected_values['insurance_provider_id'] = f"{provider_id} - {provider_name}"
            elif "Aquilana" in text and 'insurance_provider_id' not in detected_values:
                if idx > 0 and "0032" in results[idx - 1][1]:
                    provider_id = results[idx - 1][1].strip()
                    detected_values['insurance_provider_id'] = f"{provider_id} - {text}"
        
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
    
    potential_names.sort(key=lambda x: x[1])
    
    if len(potential_names) >= 2:
        detected_values['surname'] = potential_names[0][0]
        detected_values['first_name'] = potential_names[1][0]
    elif len(potential_names) == 1:
        name_text = potential_names[0][0]
        for _, label_text, _ in results:
            if "3. Name" in label_text and abs(potential_names[0][1] - y_min) < 50:
                detected_values['surname'] = name_text
                break
            elif "4. Vornamen" in label_text and abs(potential_names[0][1] - y_min) < 50:
                detected_values['first_name'] = name_text
                break
        if 'surname' not in detected_values and 'first_name' not in detected_values:
            detected_values['surname'] = name_text
    
    if 'insurance_number' in detected_values:
        card_info.insurance_number = detected_values['insurance_number']
    if 'surname' in detected_values:
        card_info.surname = detected_values['surname']
    if 'first_name' in detected_values:
        card_info.first_name = detected_values['first_name']
    if 'birth_date' in detected_values:
        card_info.birth_date = detected_values['birth_date']
    if 'personal_number' in detected_values:
        card_info.personal_number = detected_values['personal_number']
    if 'insurance_provider_id' in detected_values:
        card_info.insurance_provider_id = detected_values['insurance_provider_id']
    if 'card_number' in detected_values:
        card_info.card_number = detected_values['card_number']
    if 'expiry_date' in detected_values:
        card_info.expiry_date = detected_values['expiry_date']
    
    return card_info

def process_single_image(image_path):
    try:
        output_dir = "detected_results"
        json_dir = "card_data"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        print(f"\nProcessing: {os.path.basename(image_path)}")
        image = cv2.imread(image_path)
        
        max_dimension = 1000
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        enhanced = enhance_image(image)
        
        reader = easyocr.Reader(['de'], gpu=False)
        results = reader.readtext(enhanced, min_size=15, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./-')
        
        if not results:
            print(f"No text detected in {image_path}")
            return
        
        card_info = extract_card_info(results)
        
        json_filename = f"card_data_{os.path.splitext(os.path.basename(image_path))[0]}.json"
        json_path = os.path.join(json_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(card_info.to_dict(), f, ensure_ascii=False, indent=2)
        
        annotated = image.copy()
        for (bbox, text, prob) in results:
            if prob > 0.5:
                points = np.array(bbox).astype(np.int32)
                cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
                x, y = points[0]
                cv2.putText(annotated, f"{text}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
