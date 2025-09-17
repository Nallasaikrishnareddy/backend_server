# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import traceback
import logging
import sys

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.DEBUG,  # Show debug/info/warning/error
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------- Imports --------------------
try:
    from embeddings import get_embedding_from_bytes, emb_to_bytes, bytes_to_emb
    from db import init_db, insert_face, find_best_match
    logger.info("✅ All imports successful")
except Exception as e:
    logger.exception(f"❌ Import error: {e}")

# -------------------- App Setup --------------------
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

# -------------------- Database Init --------------------
try:
    init_db()
    logger.info("✅ Database initialized successfully")
except Exception as e:
    logger.exception(f"❌ Database initialization failed: {e}")

# -------------------- Middleware to log requests --------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# -------------------- Health Endpoints --------------------
@app.get("/")
async def root():
    logger.debug("Health check endpoint called")
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
    try:
        init_db()
        db_status = "✅ OK"
    except Exception as e:
        db_status = f"❌ Error: {str(e)}"
    logger.debug(f"Health check: database status: {db_status}")
    return {
        "status": "running",
        "database": db_status,
        "message": "Face Recognition Backend"
    }

# -------------------- Register Endpoint --------------------
@app.post('/register')
async def register(name: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"Register request received for name: {name}")
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        logger.debug(f"Received image bytes: {len(image_bytes)}")
        if len(image_bytes) == 0:
            logger.warning("Empty image file received")
            raise HTTPException(status_code=400, detail="Empty image file")
        
        emb = get_embedding_from_bytes(image_bytes)
        logger.debug(f"Embedding generated: shape={emb.shape}, dtype={emb.dtype}")
        emb_blob = emb_to_bytes(emb)
        
        row_id = insert_face(name, emb_blob, image_bytes)
        logger.info(f"Face registered successfully for {name}, row_id={row_id}")
        
        return JSONResponse({
            'status': 'success', 
            'message': f'Face registered successfully for {name}',
            'id': row_id
        })
        
    except Exception as e:
        logger.exception(f"Registration failed for {name}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# -------------------- Verify Endpoint --------------------
@app.post('/verify')
async def verify(file: UploadFile = File(...)):
    logger.info("Verify request received")
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        logger.debug(f"Received image bytes: {len(image_bytes)}")
        if len(image_bytes) == 0:
            logger.warning("Empty image file received")
            raise HTTPException(status_code=400, detail="Empty image file")
        
        emb = get_embedding_from_bytes(image_bytes)
        match = find_best_match(emb)
        
        if match:
            logger.info(f"Match found: {match['name']} (id={match['id']})")
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
            logger.info("No matching face found")
            return JSONResponse({
                'status': 'success',
                'match_found': False,
                'match': None,
                'message': 'No matching face found'
            })
            
    except Exception as e:
        logger.exception("Verification failed")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

# -------------------- Render Deployment --------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
