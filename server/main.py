from fastapi import FastAPI, UploadFile, File, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
import time
from PIL import Image, UnidentifiedImageError

try:
    from .contract import (
        ALLOWED_IMAGE_MIME_TYPES,
        DETECTOR_STATUS,
        DETECTOR_VERSION,
        LOCALIZATION_METHOD,
        detector_metadata,
        cors_origins,
        detection_rate_limit,
        max_image_bytes,
        validate_image_headers,
        validate_target,
    )
    from .locate_model import locate_object, is_available as locate_available
    from .model import predict, INDOOR_OBJECT_ONTOLOGY as INDOOR_OBJECTS
    from .text_model import classify_text, retrain
    from .tea_dataset import TEA_TYPES, FLAVOR_LABELS, QUALITY_TIERS
except ImportError:  # Supports `uvicorn main:app` from the server directory.
    from contract import (
        ALLOWED_IMAGE_MIME_TYPES,
        DETECTOR_STATUS,
        DETECTOR_VERSION,
        LOCALIZATION_METHOD,
        detector_metadata,
        cors_origins,
        detection_rate_limit,
        max_image_bytes,
        validate_image_headers,
        validate_target,
    )
    from locate_model import locate_object, is_available as locate_available
    from model import predict, INDOOR_OBJECT_ONTOLOGY as INDOOR_OBJECTS
    from text_model import classify_text, retrain
    from tea_dataset import TEA_TYPES, FLAVOR_LABELS, QUALITY_TIERS

app = FastAPI(title="Pulse Point Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)

_DETECT_REQUESTS = {}
_MAX_IMAGE_PIXELS = 25_000_000


def _client_ip(request):
    # Do not trust a caller-controlled X-Forwarded-For header. A deployment
    # that terminates trusted proxy headers should normalize the client IP at
    # that proxy before forwarding to this service.
    return request.client.host if request.client else "unknown"


def _check_detection_rate_limit(client_ip):
    now = time.monotonic()
    recent = [stamp for stamp in _DETECT_REQUESTS.get(client_ip, []) if now - stamp < 60]
    if len(_DETECT_REQUESTS) > 10_000:
        for known_ip, stamps in list(_DETECT_REQUESTS.items()):
            if not stamps or now - stamps[-1] >= 60:
                _DETECT_REQUESTS.pop(known_ip, None)
    if len(recent) >= detection_rate_limit():
        _DETECT_REQUESTS[client_ip] = recent
        return False
    recent.append(now)
    _DETECT_REQUESTS[client_ip] = recent
    return True


def _validate_decodable_image(image_bytes, content_type):
    """Verify format and dimensions before passing bytes into the ML stack."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            expected_formats = {
                "image/jpeg": {"JPEG"},
                "image/png": {"PNG"},
                "image/webp": {"WEBP"},
            }
            if image.format not in expected_formats[content_type]:
                raise ValueError("image content does not match its MIME type")
            if image.width <= 0 or image.height <= 0:
                raise ValueError("image dimensions must be positive")
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise ValueError("image dimensions are too large")
            image.verify()
    except (KeyError, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError("malformed or unsupported image") from error


@app.get("/health")
async def health():
    metadata = detector_metadata()
    metadata.update({
        "detectorVersion": DETECTOR_VERSION,
        "status": DETECTOR_STATUS,
        "localizationMethod": LOCALIZATION_METHOD,
        "supportedObjectOntology": sorted(INDOOR_OBJECTS.keys()),
    })
    return {
        "status": "ok",
        "apiVersion": "v1",
        "objects": len(INDOOR_OBJECTS),
        "detector": metadata,
        "locateAnything": locate_available(),
        "fallbackObjects": len(INDOOR_OBJECTS),
    }


@app.get("/objects")
async def list_objects():
    return {"objects": sorted(INDOOR_OBJECTS.keys())}


@app.post("/detect")
@app.post("/v1/detect")
async def detect(
    request: Request,
    image: UploadFile = File(...),
    target: str = Form(default=""),
):
    if not _check_detection_rate_limit(_client_ip(request)):
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": "60"},
            content={"error": "Too many detection requests. Try again in a minute."},
        )

    try:
        normalized_target = validate_target(target)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})

    content_type = (image.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_MIME_TYPES:
        return JSONResponse(
            status_code=415,
            content={"error": "Unsupported image MIME type. Use JPEG, PNG, or WebP."},
        )

    declared_size = request.headers.get("content-length")
    if declared_size:
        try:
            # Allow modest multipart overhead while rejecting obviously
            # oversized bodies before UploadFile is read.
            if int(declared_size) > max_image_bytes() + 1_048_576:
                return JSONResponse(status_code=413, content={"error": "Image too large."})
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid Content-Length header."})

    start = time.time()
    image_bytes = await image.read(max_image_bytes() + 1)

    try:
        validate_image_headers(content_type, len(image_bytes))
    except OverflowError:
        return JSONResponse(status_code=413, content={"error": "Image too large."})
    except ValueError as error:
        return JSONResponse(status_code=400, content={"error": str(error)})

    try:
        _validate_decodable_image(image_bytes, content_type)
    except ValueError as error:
        return JSONResponse(status_code=400, content={"error": str(error)})

    result = None

    # ── Primary: LocateAnything-3B (open-vocabulary, any target) ──
    if normalized_target:
        result = locate_object(image_bytes, normalized_target)

    # ── Fallback: PulsePointNet (indoor-object ontology, ~45 classes) ──
    if result is None:
        try:
            result = predict(image_bytes, target_name=normalized_target or None)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            return JSONResponse(status_code=400, content={"error": "Malformed image."})

    # Keep the safety boundary authoritative even if a model adapter (either
    # the PulsePointNet fallback or the LocateAnything-3B primary path)
    # returns incomplete or overly optimistic metadata.
    response_metadata = detector_metadata()
    if isinstance(result.get("metadata"), dict):
        response_metadata.update(result["metadata"])
    response_metadata.update({
        "assistiveReady": False,
        "proof": False,
        "assistiveReadyProof": False,
    })
    result["metadata"] = response_metadata
    result["assistiveReady"] = False
    result["assistiveReadyProof"] = False
    result["proof"] = False
    result["latency_ms"] = round((time.time() - start) * 1000)
    return result


# ── Text / Tea CNN endpoints ─────────────────────────────────────────────────

@app.post("/classify-text")
async def classify_text_endpoint(
    text: str = Body(..., embed=True, description="Tea description or spoken query"),
    top_k: int = Body(3, embed=True, description="Number of alternative tea types to return"),
):
    if not text or not text.strip():
        return JSONResponse(status_code=422, content={"error": "text must be non-empty"})
    if len(text) > 512:
        return JSONResponse(status_code=422, content={"error": "text exceeds 512 characters"})

    start = time.time()
    result = classify_text(text.strip(), top_k=top_k)
    result["latency_ms"] = round((time.time() - start) * 1000)
    result["input"] = text.strip()
    return result


@app.get("/tea-schema")
async def tea_schema():
    return {
        "tea_types":     TEA_TYPES,
        "flavor_labels": FLAVOR_LABELS,
        "quality_tiers": QUALITY_TIERS,
    }


@app.post("/retrain-text")
async def retrain_text():
    start = time.time()
    info = retrain()
    info["duration_ms"] = round((time.time() - start) * 1000)
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
