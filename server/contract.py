"""Small, dependency-free pieces of the Pulse Point HTTP contract.

This module intentionally does not import torch or PIL.  It keeps boundary
tests useful in environments that do not have the detector's ML stack.
"""

import os
import re


API_VERSION = "v1"
DETECTOR_VERSION = "pulsepoint-imagenet-gradcam-experimental-v1"
DETECTOR_STATUS = "experimental"
VALIDATION_STATUS = "unvalidated"
CONFIDENCE_CALIBRATION = "uncalibrated"
LOCALIZATION_METHOD = "approximate/Grad-CAM"
ASSISTIVE_READY = False
PROOF = False

DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES_HARD_LIMIT = 20 * 1024 * 1024
MAX_TARGET_LENGTH = 64
ALLOWED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})

DEFAULT_CORS_ORIGINS = (
    "https://pulse-point-steel.vercel.app",
    "http://localhost:5173",
    "http://localhost:4173",
)


def _bounded_int_env(name, default, *, minimum, maximum):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(value, maximum))


def max_image_bytes():
    """Return a bounded upload limit; configuration cannot exceed the hard cap."""

    return _bounded_int_env(
        "PULSEPOINT_MAX_IMAGE_BYTES",
        DEFAULT_MAX_IMAGE_BYTES,
        minimum=1,
        maximum=MAX_IMAGE_BYTES_HARD_LIMIT,
    )


def detection_rate_limit():
    """Requests per IP per minute for the experimental detector endpoint."""

    return _bounded_int_env(
        "PULSEPOINT_DETECT_RATE_LIMIT",
        30,
        minimum=1,
        maximum=120,
    )


def cors_origins():
    """Return explicit CORS origins, failing closed for wildcard/invalid values."""

    configured = os.getenv("PULSEPOINT_CORS_ORIGINS")
    if configured is None:
        return list(DEFAULT_CORS_ORIGINS)

    origins = []
    for value in configured.split(","):
        origin = value.strip().rstrip("/")
        if not origin or origin == "*":
            continue
        if re.fullmatch(r"https?://[^/\s]+", origin):
            origins.append(origin)
    return origins


def validate_target(target):
    """Normalize an optional target and reject control characters/oversized input."""

    if target is None:
        return ""
    if not isinstance(target, str):
        raise ValueError("target must be text")
    normalized = target.strip()
    if len(normalized) > MAX_TARGET_LENGTH:
        raise ValueError(f"target exceeds {MAX_TARGET_LENGTH} characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("target contains unsupported control characters")
    return normalized


def validate_image_headers(content_type, byte_count):
    """Validate cheap request properties before image decoding."""

    if content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError("unsupported image MIME type")
    if byte_count <= 0:
        raise ValueError("image must not be empty")
    if byte_count > max_image_bytes():
        raise OverflowError(f"image exceeds {max_image_bytes()} bytes")


def detector_metadata():
    """Return a fresh metadata object so callers cannot mutate the constants."""

    return {
        "apiVersion": API_VERSION,
        "modelName": "PulsePointNet",
        "modelVersion": DETECTOR_VERSION,
        "detectorVersion": DETECTOR_VERSION,
        "status": DETECTOR_STATUS,
        "validationStatus": VALIDATION_STATUS,
        "confidenceCalibration": CONFIDENCE_CALIBRATION,
        "localizationMethod": LOCALIZATION_METHOD,
        "assistiveReady": ASSISTIVE_READY,
        "proof": PROOF,
        "assistiveReadyProof": PROOF,
    }
