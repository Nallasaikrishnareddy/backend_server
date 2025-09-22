# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException , Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import base64
from embeddings import get_embedding_from_bytes, emb_to_bytes, bytes_to_emb
from db import init_db, insert_face, find_best_match
from pydantic import BaseModel


app = FastAPI(
    title="Face Recognition API",
    description="Face registration and verification system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


init_db()

class RegisterPayload(BaseModel):
    name: str
    image: str

class VerifyPayload(BaseModel):
    image_base64: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Face Recognition API is running",
        "status": "healthy",
        "endpoints": {
            "register": "POST /register",
            "verify": "POST /verify"
        }
    }

@app.post('/register')
async def register(payload: RegisterPayload = Body(...)):
    try:
        try:
            print("DEBUG: Decoding image...")
            image_bytes = base64.b64decode(payload.image)
            print(f"DEBUG: Image decoded, size={len(image_bytes)}")
        except Exception as decode_error:
            print(f"ERROR: Base64 decode failed -> {decode_error}")
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        if len(image_bytes) == 0:
            print("WARNING: Empty image data after decoding")
            raise HTTPException(status_code=400, detail="Empty image data")

        print("DEBUG: Getting embedding...")
        emb = get_embedding_from_bytes(image_bytes)
        print(f"DEBUG: Embedding length={len(emb)}")

        print("DEBUG: Converting embedding to blob...")
        emb_blob = emb_to_bytes(emb)
        print(f"DEBUG: Blob size={len(emb_blob)}")

        print(f"DEBUG: Inserting face record for {payload.name}...")
        row_id = insert_face(payload.name, emb_blob, image_bytes)
        print(f"INFO: Insert successful, row_id={row_id}")

        return JSONResponse({
            'status': 'success',
            'message': f'Face registered successfully for {payload.name}',
            'id': row_id
        })

    except HTTPException as http_err:
        print(f"HTTPException: {http_err.detail}")
        raise
    except Exception as e:
        print(f"EXCEPTION: Unexpected error -> {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post('/verify')
async def verify(payload: VerifyPayload = Body(...)):
    try:
        print("DEBUG: Starting verification...")

        print("DEBUG: Decoding image from base64...")
        image_bytes = base64.b64decode(payload.image_base64)
        print(f"DEBUG: Image decoded, size={len(image_bytes)}")

        if len(image_bytes) == 0:
            print("WARNING: Empty image data after decoding")
            raise HTTPException(status_code=400, detail="Empty image data")

        print("DEBUG: Extracting embedding from image bytes...")
        emb = get_embedding_from_bytes(image_bytes)
        print(f"DEBUG: Embedding extracted, length={len(emb)}")

        print("DEBUG: Searching for best match in database...")
        match = find_best_match(emb)

        if match:
            print(f"INFO: Match found -> id={match['id']}, name={match['name']}, score={match['score']}")
            return JSONResponse({
                'status': 'success',
                'match_found': True,
                'match': {
                    'id': match['id'],
                    'name': match['name'],
                    'confidence': round(match['score'], 4)
                }
            })
        else:
            print("INFO: No matching face found")
            return JSONResponse({
                'status': 'success',
                'match_found': False,
                'match': None,
                'message': 'No matching face found'
            })

    except HTTPException as http_err:
        print(f"HTTPException: {http_err.detail}")
        raise
    except Exception as e:
        print(f"EXCEPTION: Unexpected error during verification -> {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# # For Render deployment
# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
