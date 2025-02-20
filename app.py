from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from PIL import Image
import io
import cv2
import numpy as np
from ocr_reader import extract_card_info, enhance_image
import easyocr
import json
import base64

app = FastAPI(
    title="Health Card OCR API",
    description="API for extracting information from European Health Insurance Cards",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the OCR reader once
reader = easyocr.Reader(['de'])

@app.post("/api/extract-card-info")
async def extract_info(file: UploadFile = File(...)):
    """
    Extract information from a health insurance card image and return both data and annotated image
    """
    try:
        # Read and validate the image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Resize image if too large
        max_dimension = 1500
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Enhance image
        enhanced = enhance_image(image)
        
        # Perform OCR
        results = reader.readtext(enhanced)
        
        if not results:
            raise HTTPException(status_code=400, detail="No text detected in image")
        
        # Create annotated image
        annotated = image.copy()
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Only show confident detections
                points = np.array(bbox).astype(np.int32)
                cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
                x, y = points[0]
                cv2.putText(annotated, f"{text}", (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract card information
        card_info = extract_card_info(results)
        
        # Return both the extracted information and annotated image
        return {
            "card_data": card_info.to_dict(),
            "annotated_image": annotated_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 
