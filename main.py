# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import traceback

# Add error handling for imports
try:
    from embeddings import get_embedding_from_bytes, emb_to_bytes, bytes_to_emb
    from db import init_db, insert_face, find_best_match
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    print(traceback.format_exc())

app = FastAPI(
    title="Face Recognition API",
    description="Face registration and verification system",
    version="1.0.0"
)

# CORS configuration - MORE PERMISSIVE FOR TESTING
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup with error handling
try:
    init_db()
    print("✅ Database initialized successfully")
except Exception as e:
    print(f"❌ Database initialization failed: {e}")

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

@app.get("/health")
async def health():
    """Detailed health check"""
    try:
        # Test database connection
        init_db()
        db_status = "✅ OK"
    except Exception as e:
        db_status = f"❌ Error: {str(e)}"
    
    return {
        "status": "running",
        "database": db_status,
        "message": "Face Recognition Backend"
    }

@app.post('/register')
async def register(name: str = Form(...), file: UploadFile = File(...)):
    """Register a new face with name and image"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Get embedding
        emb = get_embedding_from_bytes(image_bytes)  # numpy array float32
        emb_blob = emb_to_bytes(emb)
        
        # Store in database
        row_id = insert_face(name, emb_blob, image_bytes)
        
        return JSONResponse({
            'status': 'success', 
            'message': f'Face registered successfully for {name}',
            'id': row_id
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post('/verify')
async def verify(file: UploadFile = File(...)):
    """Verify a face against registered faces"""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Get embedding
        emb = get_embedding_from_bytes(image_bytes)
        
        # Find match
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
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# For Render deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)