# embeddings.py
import io
import zlib
import numpy as np
from PIL import Image
import tempfile
import os

# -------------------- Lazy model holder --------------------
_deepface_model = None
_deepface_backend_name = "ArcFace"  # You can switch to "MobileFaceNet" if memory is tight

def _get_model():
    """
    Load DeepFace model on first use (lazy loading).
    Reduces memory spike on Render free tier.
    """
    global _deepface_model
    if _deepface_model is None:
        try:
            from deepface import DeepFace
            print("[INFO] Loading DeepFace model on-demand...")
            _deepface_model = DeepFace.build_model(_deepface_backend_name)
            print("[INFO] DeepFace model loaded successfully")
        except Exception as e:
            raise RuntimeError("Failed to load DeepFace model") from e
    return _deepface_model

# -------------------- Embedding functions --------------------
def get_embedding_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to a normalized embedding (float32) using DeepFace.
    Optimized for cloud deployment (on-demand model + temp files fallback).
    """
    from deepface import DeepFace

    # Load model lazily
    model = _get_model()

    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    rep = None
    try:
        # Fast path: numpy array
        rep = DeepFace.represent(
            img_path=np.array(img),
            model_name=_deepface_backend_name,
            model=model,  # Pass already loaded model
            enforce_detection=True
        )
    except Exception as e:
        print(f"[WARN] Numpy array input failed, using temp file fallback: {e}")
        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            img.save(tmp_file, format="JPEG")
            tmp_path = tmp_file.name
        try:
            rep = DeepFace.represent(
                img_path=tmp_path,
                model_name=_deepface_backend_name,
                model=model,
                enforce_detection=True
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Parse embedding
    emb = None
    if isinstance(rep, list) and len(rep) > 0 and "embedding" in rep[0]:
        emb = np.array(rep[0]["embedding"], dtype=np.float32)
    elif isinstance(rep, dict) and "embedding" in rep:
        emb = np.array(rep["embedding"], dtype=np.float32)

    if emb is None:
        raise RuntimeError("DeepFace did not return an embedding.")

    # Normalize
    norm = np.linalg.norm(emb)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding from DeepFace.")

    return (emb / norm).astype(np.float32)

def emb_to_bytes(emb: np.ndarray) -> bytes:
    """
    Compress and convert embedding to bytes for DB storage (float16 + zlib).
    """
    f16 = emb.astype(np.float16)
    raw = f16.tobytes()
    compressed = zlib.compress(raw, level=6)
    return compressed

def bytes_to_emb(blob: bytes) -> np.ndarray:
    """
    Decompress bytes back to float32 numpy array.
    """
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float16)
    return arr.astype(np.float32)
