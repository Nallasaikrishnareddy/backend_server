# embeddings.py
import io
import zlib
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

# Lazy model holder
_insightface_app = None


def _get_app():
    """
    Initialize InsightFace ArcFace model (buffalo_l).
    Uses ONNXRuntime backend (lightweight, CPU-friendly).
    """
    global _insightface_app
    if _insightface_app is None:
        print("[INFO] Loading InsightFace model (buffalo_l)...")
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1)  # CPU mode (works on Render free tier)
        _insightface_app = app
        print("[INFO] InsightFace model loaded successfully.")
    return _insightface_app


def get_embedding_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to a normalized ArcFace embedding (float32) using InsightFace.
    """
    app = _get_app()

    # Load image
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)

    # Detect and get embeddings
    faces = app.get(img_np)
    if len(faces) == 0:
        raise RuntimeError("No face detected in the image.")

    emb = faces[0].embedding.astype(np.float32)

    # Normalize
    norm = np.linalg.norm(emb)
    if norm == 0:
        raise RuntimeError("Zero-norm embedding from InsightFace.")
    return (emb / norm).astype(np.float32)


def emb_to_bytes(emb: np.ndarray) -> bytes:
    """Compress and convert embedding to bytes for DB storage (float16 + zlib)."""
    f16 = emb.astype(np.float16)
    raw = f16.tobytes()
    compressed = zlib.compress(raw, level=6)
    return compressed


def bytes_to_emb(blob: bytes) -> np.ndarray:
    """Decompress bytes back to float32 numpy array."""
    raw = zlib.decompress(blob)
    arr = np.frombuffer(raw, dtype=np.float16)
    return arr.astype(np.float32)
