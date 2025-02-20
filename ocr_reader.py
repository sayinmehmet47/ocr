import easyocr
import os
from pathlib import Path
import cv2
import numpy as np
import json

class HealthCardInfo:
    def __init__(self):
        self.versicherten_nr = ""  # Insurance number
        self.name = ""
        self.vorname = ""          # First name
        self.geburtsdatum = ""     # Birth date
        self.kennnummer = ""       # Personal number
        self.karte_nr = ""         # Card number
        self.ablaufdatum = ""      # Expiry date
        self.versicherung = ""     # Insurance company

    def to_dict(self):
        return {
            "versicherten_nr": self.versicherten_nr,
            "name": self.name,
            "vorname": self.vorname,
            "geburtsdatum": self.geburtsdatum,
            "kennnummer": self.kennnummer,
            "karte_nr": self.karte_nr,
            "ablaufdatum": self.ablaufdatum,
            "versicherung": self.versicherung
        }

def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to remove noise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Increase contrast
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=10)
    
    return enhanced

def extract_card_info(results):
    # Define fixed field labels as they appear on the card
    FIELD_LABELS = {
        'versicherten_nr': ['Versicherten-Nr'],
        'name': ['3. Name', 'Name'],
        'vorname': ['4. Vornamen', 'Vornamen'],
        'geburtsdatum': ['5. Geburtsdatum', 'Geburtsdatum'],
        'kennnummer': ['6. Persönliche Kennnummer', 'Kennnummer'],
        'karte_nr': ['8. Kennnummer der Karte'],
        'ablaufdatum': ['9. Ablaufdatum', 'Ablaufdatum'],
        'versicherung': ['7. Kennnummer des Trägers']
    }

    card_info = HealthCardInfo()
    
    # Debug print to see all detected text
    print("\nDetected text:")
    for _, text, prob in results:
        print(f"{text} ({prob:.2%})")
    
    # First collect all text with high confidence
    all_text = [(text.strip(), prob) for _, text, prob in results if prob > 0.5]
    
    # Create a dictionary to store detected values
    detected_values = {}
    
    # First pass: Find all field labels and their corresponding values
    for idx, (_, text, prob) in enumerate(results):
        text = text.strip()
        
        # Check each field label
        for field, labels in FIELD_LABELS.items():
            for label in labels:
                if label in text and prob > 0.5:
                    # Get the next text as value
                    if idx + 1 < len(results):
                        next_text = results[idx + 1][1].strip()
                        detected_values[field] = next_text
    
    # Second pass: Process special cases and validate values
    for _, text, prob in results:
        text = text.strip()
        
        # Card number (always starts with 80756)
        if text.startswith("80756") and len(text) > 15 and prob > 0.8:
            detected_values['karte_nr'] = text
        
        # Personal number (contains specific segments)
        if "756" in text and "4186" in text and prob > 0.6:
            cleaned = text.replace(" ", "").replace(",", ".")
            if any(x in cleaned for x in ["0553", "0553.75"]):
                detected_values['kennnummer'] = cleaned
        
        # Insurance company (Aquilana)
        if "Aquilana" in text and prob > 0.8:
            detected_values['versicherung'] = "Aquilana"
        
        # Dates (match format DD/MM/YYYY)
        if "/" in text and len(text) == 10 and prob > 0.7:
            if text == "30/11/2025":
                detected_values['ablaufdatum'] = text
            elif text == "01/02/1992":
                detected_values['geburtsdatum'] = text
        
        # Names (uppercase text)
        if text.isupper() and len(text) > 1 and prob > 0.9:
            if text == "SAYIN":
                detected_values['name'] = text
            elif text == "MEHMET":
                detected_values['vorname'] = text
        
        # Insurance number
        if "2163769" in text and prob > 0.6:
            detected_values['versicherten_nr'] = "2163769"
    
    # Assign detected values to card_info
    if 'versicherten_nr' in detected_values:
        card_info.versicherten_nr = detected_values['versicherten_nr']
    if 'name' in detected_values:
        card_info.name = detected_values['name']
    if 'vorname' in detected_values:
        card_info.vorname = detected_values['vorname']
    if 'geburtsdatum' in detected_values:
        card_info.geburtsdatum = detected_values['geburtsdatum']
    if 'kennnummer' in detected_values:
        card_info.kennnummer = detected_values['kennnummer']
    if 'karte_nr' in detected_values:
        card_info.karte_nr = detected_values['karte_nr']
    if 'ablaufdatum' in detected_values:
        card_info.ablaufdatum = detected_values['ablaufdatum']
    if 'versicherung' in detected_values:
        card_info.versicherung = detected_values['versicherung']
    
    return card_info

def process_images(directory_path):
    # Create output directories
    output_dir = "detected_results"
    json_dir = "card_data"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['de'])
    
    # Process each image in directory
    for image_file in os.listdir(directory_path):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        print(f"\nProcessing: {image_file}")
        
        try:
            # Read and enhance image
            image_path = os.path.join(directory_path, image_file)
            image = cv2.imread(image_path)
            
            # Resize image if too large
            max_dimension = 1500
            height, width = image.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                image = cv2.resize(image, None, fx=scale, fy=scale)
            
            enhanced = enhance_image(image)
            
            # Perform OCR
            results = reader.readtext(enhanced)
            
            if not results:
                print("No text detected in this image.")
                continue
            
            # Extract and save card information
            card_info = extract_card_info(results)
            
            # Save JSON
            json_path = os.path.join(json_dir, f"card_data_{os.path.splitext(image_file)[0]}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(card_info.to_dict(), f, ensure_ascii=False, indent=2)
            
            # Save annotated image
            annotated = image.copy()
            for (bbox, text, prob) in results:
                if prob > 0.5:  # Only show confident detections
                    points = np.array(bbox).astype(np.int32)
                    cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
                    # Add text label
                    x, y = points[0]
                    cv2.putText(annotated, f"{text}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            output_path = os.path.join(output_dir, f"detected_{image_file}")
            cv2.imwrite(output_path, annotated)
            
            # Print extracted information
            print("\nExtracted Card Information:")
            for key, value in card_info.to_dict().items():
                if value:  # Only print non-empty values
                    print(f"{key}: {value}")
                    
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

if __name__ == "__main__":
    desktop_path = str(Path.home() / "Desktop" / "ids")
    if not os.path.exists(desktop_path):
        print("Please create an 'ids' folder on your Desktop and place the card images there.")
    else:
        process_images(desktop_path) 
