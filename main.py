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
            image_bytes = base64.b64decode(payload.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image data")
        emb = get_embedding_from_bytes(image_bytes)
        emb_blob = emb_to_bytes(emb)
        row_id = insert_face(payload.name, emb_blob, image_bytes)
        return JSONResponse({
            'status': 'success',
            'message': f'Face registered successfully for {payload.name}',
            'id': row_id
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post('/verify')
async def verify(payload: VerifyPayload = Body(...) ):
    try:
        image_bytes = base64.b64decode(payload.image_base64)
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image data")
        emb = get_embedding_from_bytes(image_bytes)
        match = find_best_match(emb)

        if match:
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
            return JSONResponse({
                'status': 'success',
                'match_found': False,
                'match': None,
                'message': 'No matching face found'
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)