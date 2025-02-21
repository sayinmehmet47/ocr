from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from ocr_reader import process_single_image, enhance_image, extract_card_info
import easyocr
import tempfile
import os
from pathlib import Path
import base64
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Health Insurance Card OCR API",
    description="API for extracting information from German health insurance cards",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def encode_image_to_base64(image):
    """Convert an OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.get("/")
async def root():
    return {"message": "Welcome to the Health Insurance Card OCR API"}

@app.post("/process-card/")
async def process_card(
    file: UploadFile = File(description="Health insurance card image file")
):
    try:
        logger.debug(f"Received request for file processing")
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
            
        logger.debug(f"Received file: {file.filename}")
        logger.debug(f"Content type: {file.content_type}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image file
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file received")
            
        logger.debug(f"File size: {len(contents)} bytes")
        
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        logger.debug(f"Image shape: {image.shape}")

        # Process image
        max_dimension = 1000
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # Store original image as base64
        original_image_base64 = encode_image_to_base64(image)
        
        # Enhance image
        enhanced = enhance_image(image)
        
        # Initialize OCR reader
        reader = easyocr.Reader(['de'], gpu=False)
        results = reader.readtext(enhanced, min_size=15, 
                                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./-')
        
        if not results:
            return JSONResponse(
                status_code=404,
                content={"error": "No text detected in the image"}
            )
        
        logger.debug(f"Number of text regions detected: {len(results)}")
        
        # Extract card information
        card_info = extract_card_info(results)
        
        # Create annotated image
        annotated = image.copy()
        for (bbox, text, prob) in results:
            if prob > 0.5:
                points = np.array(bbox).astype(np.int32)
                cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
                x, y = points[0]
                cv2.putText(annotated, f"{text}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert annotated image to base64
        annotated_image_base64 = encode_image_to_base64(annotated)
            
        # Return the extracted information with images
        response_data = {
            "status": "success",
            "card_info": card_info.to_dict(),
            "confidence_scores": {
                text: f"{prob:.2%}" for _, text, prob in results if prob > 0.5
            },
            "images": {
                "original": original_image_base64,
                "annotated": annotated_image_base64
            },
            "processing_details": {
                "original_size": f"{width}x{height}",
                "processed_size": f"{image.shape[1]}x{image.shape[0]}",
                "file_name": file.filename,
                "content_type": file.content_type
            }
        }
        
        logger.debug("Successfully processed image and generated response")
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred while processing the image: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug") 
