from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
from ocr_reader import extract_card_info, process_image_ocr
from utils import (
    enhance_image,
    resize_image_if_needed,
    encode_image_to_base64,
    create_annotated_image,
    SUPPORTED_LANGUAGES,
    logger
)

app = FastAPI(
    title="Health Insurance Card OCR API",
    description="API for extracting information from German, French, and Italian health insurance cards",
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

@app.get("/")
async def root():
    return {
        "message": "Health Insurance Card OCR API",
        "supported_languages": SUPPORTED_LANGUAGES
    }

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
        height, width = image.shape[:2]
        image = resize_image_if_needed(image)
        
        # Store original image as base64
        original_image_base64 = encode_image_to_base64(image)
        
        # Process image with OCR
        results = process_image_ocr(image)
        
        if not results:
            return JSONResponse(
                status_code=404,
                content={"error": "No text detected in the image"}
            )
        
        logger.debug(f"Number of text regions detected: {len(results)}")
        
        # Extract card information
        card_info = extract_card_info(results)
        
        # Create annotated image
        annotated = create_annotated_image(image, results)
        annotated_image_base64 = encode_image_to_base64(annotated)
            
        # Return the extracted information with images
        response_data = {
            "status": "success",
            "card_info": card_info.to_dict(),
            "confidence_scores": {
                result[1][0]: f"{result[1][1]:.2%}" for result in results if result[1][1] > 0.5
            },
            "images": {
                "original": original_image_base64,
                "annotated": annotated_image_base64
            },
            "processing_details": {
                "original_size": f"{width}x{height}",
                "processed_size": f"{image.shape[1]}x{image.shape[0]}",
                "file_name": file.filename,
                "content_type": file.content_type,
                "detected_language": card_info.detected_language
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
