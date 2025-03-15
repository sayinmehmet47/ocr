import cv2
import numpy as np
import base64
import logging
from PIL import Image, ImageDraw, ImageFont
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Multi-language field labels
FIELD_LABELS = {
    'insurance_number': {
        'de': ['Versicherten-Nr', 'Versicherten-Nr.', 'ersicherten-Nr', 'ersicherten-Nr.'],
        'fr': ['N° d\'assuré', 'Numéro d\'assuré', 'N° assuré', 'No. d\'assuré'],
        'it': ['N. assicurato', 'Numero assicurato', 'N° assicurato']
    },
    'surname': {
        'de': ['3. Name', 'Name', '3.Name'],
        'fr': ['3. Nom', 'Nom', '3.Nom'],
        'it': ['3. Cognome', 'Cognome', '3.Cognome']
    },
    'first_name': {
        'de': ['4. Vornamen', 'Vornamen', '4.Vornamen'],
        'fr': ['4. Prénoms', 'Prénoms', '4.Prénoms'],
        'it': ['4. Nome', 'Nome', '4.Nome']
    },
    'birth_date': {
        'de': ['5. Geburtsdatum', 'Geburtsdatum', '5.Geburtsdatum', 'Geburtsdat'],
        'fr': ['5. Date de naissance', 'Date de naissance', '5.Date de naissance'],
        'it': ['5. Data di nascita', 'Data di nascita', '5.Data di nascita']
    },
    'personal_number': {
        'de': ['6. Persönliche Kennnummer', 'Kennnummer', '6. Personliche'],
        'fr': ['6. Numéro personnel', 'Numéro personnel', '6.Numéro personnel'],
        'it': ['6. Numero personale', 'Numero personale', '6.Numero personale']
    },
    'insurance_provider_id': {
        'de': ['7. Kennnummer des Trägers', '7. Kennnummer'],
        'fr': ['7. Code de l\'organisme', '7.Code organisme'],
        'it': ['7. Codice ente', '7.Codice ente']
    },
    'card_number': {
        'de': ['8. Kennnummer der Karte'],
        'fr': ['8. Numéro de la carte'],
        'it': ['8. Numero della carta']
    },
    'expiry_date': {
        'de': ['9. Ablaufdatum'],
        'fr': ['9. Date d\'expiration'],
        'it': ['9. Data di scadenza']
    }
}

# Common words to exclude from name detection
EXCLUDED_WORDS = {
    'de': ['KARTE', 'VERSICHERUNG', 'EUROPEAN', 'EUROPÄISCHE', 'EUROPAISCHE', 'EUROPÀISCHE', 'KRANKENVERSICHERUNGSKARTE'],
    'fr': ['CARTE', 'ASSURANCE', 'EUROPÉENNE', 'SANTÉ', 'MALADIE'],
    'it': ['CARTA', 'ASSICURAZIONE', 'EUROPEA', 'SANITARIA', 'MALATTIA', 'TESSERA']
}

COUNTRY_CODES = ['CH']

SUPPORTED_LANGUAGES = ['de', 'fr', 'it']

def encode_image_to_base64(image):
    """Convert an OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')
    
def enhance_image(image):
    """Enhance image for better OCR processing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.convertScaleAbs(denoised, alpha=1.5, beta=10)
    return enhanced

def create_annotated_image(image, results):
    """Create annotated image with detected text regions."""
    # Make a copy of the original image
    annotated = image.copy()
    
    # Convert OpenCV image (BGR) to PIL Image (RGB)
    image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        # Try to find a system font that supports umlauts
        font_size = 28 
        if os.path.exists('/System/Library/Fonts/Helvetica.ttc'):  # macOS
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size, index=1)  # index=1 for bold variant
        elif os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):  # Linux
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
        else:  # Fallback to default
            font = ImageFont.load_default()
    except Exception as e:
        logger.warning(f"Could not load font: {e}. Using default font.")
        font = ImageFont.load_default()

    for result in results:
        points = np.array(result[0]).astype(np.int32)
        text = result[1][0]
        prob = result[1][1]
        
        if prob > 0.5:
            # Draw bounding box on the OpenCV image with thicker line
            cv2.polylines(annotated, [points], True, (0, 255, 0), 3)
            
            # Get coordinates for text
            x, y = points[0]
            
            # Calculate text size for the frame
            text_width, text_height = draw.textsize(text, font=font)
            padding = 10  # Increased padding around text
            
            # Draw frame around text with thicker width
            draw.rectangle([
                (x, y-30 - padding),  # Adjusted y offset for better positioning
                (x + text_width + padding * 2, y-30 + text_height + padding)
            ], outline=(0, 255, 0), width=3)  # Increased outline width
            
            # Create stronger stroke effect for better readability
            stroke_color = (0, 100, 0)  # Darker green for stroke
            offsets = [
                (-1, -1), (1, -1), (-1, 1), (1, 1),  # Diagonal offsets
                (0, -1), (-1, 0), (1, 0), (0, 1),    # Cardinal offsets
            ]
            
            # Draw stroke effect
            for dx, dy in offsets:
                draw.text((x + dx, y-30 + dy), text, fill=stroke_color, font=font)
            
            # Draw the main text
            draw.text((x, y-30), text, fill=(0, 255, 0), font=font)
    
    # Convert back to OpenCV format (RGB to BGR)
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return final_image 
