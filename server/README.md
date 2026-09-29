# Pulse Point Python Service

This folder contains an optional experimental vision API and unrelated legacy tea-text classification endpoints. The vision service is not validated for assistive use, navigation, obstacle avoidance, or safety-critical decisions.

## Vision API

POST /detect and POST /v1/detect are aliases with the same handler. Requests use multipart/form-data:

- image: JPEG, PNG, or WebP; 10 MiB default upload limit, configurable up to a hard cap of 20 MiB.
- target: optional printable text, maximum 64 characters.

The handler validates the target, content type, encoded and decoded image, image dimensions, upload size, CORS origin, and per-IP request rate before inference. GET /health reports service/detector metadata and model availability; GET /objects lists the indoor-object ontology.

### Model path and limitations

When a target is supplied, the service tries LocateAnything-3B first. If that model is unavailable, errors, or returns no parsed box, the handler falls back to the older indoor-object classifier. That fallback maps ImageNet classification probabilities to a small indoor-object ontology and uses approximate Grad-CAM for localization. It is not trained or evaluated as a bounding-box detector.

The LocateAnything adapter currently returns a fixed confidence score of 0.90; this is not a calibrated probability. The fallback classifier's raw ImageNet probability is also uncalibrated. Both paths are experimental and unvalidated. Responses force assistiveReady, assistiveReadyProof, and proof to false.

The shared response metadata currently retains the legacy localizationMethod value approximate/Grad-CAM, including on a LocateAnything result. That field does not precisely describe the primary model's box-generation path. Do not present any response field as evidence of model validation or assistive readiness.

## Web Client Data Flow

The web detector is local by default when VITE_SERVER_URL is unset. If the variable is set and the service health check succeeds, the web app sends a JPEG camera frame and target text approximately every 2.5 seconds while scanning. This remote request currently runs even when local detection has a match. The local match is preferred; the returned remote box is selected only when local matching has no result. Therefore the remote result is a fallback, but the remote request is not gated on local failure.

Anyone configuring this endpoint should understand that camera imagery and target text leave the device during scanning, including while the local detector is succeeding. VITE_SERVER_URL is a public client-side build setting, not a place for secrets.

## Configuration

- PULSEPOINT_CORS_ORIGINS: comma-separated explicit HTTP/HTTPS origins. Wildcards are ignored; invalid configured values fail closed.
- PULSEPOINT_MAX_IMAGE_BYTES: upload limit, bounded to 20 MiB; default 10 MiB.
- PULSEPOINT_DETECT_RATE_LIMIT: detector requests per IP per minute, bounded from 1 to 120; default 30.
- PULSEPOINT_RAW_CONFIDENCE_THRESHOLD: raw ImageNet probability gate for the fallback classifier, default 0.50 and never accepted below 0.50. It does not calibrate the LocateAnything fixed score.

## Deployment

The repository includes a Modal deployment definition in modal_app.py that packages the API with an A10G GPU and a persistent Hugging Face cache. Deployment needs the Modal CLI/account and may incur GPU/storage charges according to the hosting account's current terms. Review those costs before deploying. The resulting service URL can be supplied as VITE_SERVER_URL to the web build, which enables the camera-frame upload flow described above.

For local development, install requirements.txt and run the FastAPI app with Uvicorn. The service may attempt to download large model weights on first use; do not assume first-start latency or CPU inference is suitable for live scanning.

The tea-text classifier routes are legacy capabilities unrelated to object detection.
