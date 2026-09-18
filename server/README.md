# Pulse Point Python service

This service contains two separate capabilities: an experimental Pulse Point
vision endpoint and the unrelated tea-text classifier endpoints. The tea
endpoints remain available for existing callers; their presence does not make
the vision detector production-ready.

## Experimental vision contract

`POST /detect` is the legacy mobile-compatible route. `POST /v1/detect` is the
versioned alias and uses the same handler and response shape.

The request is `multipart/form-data` with:

- `image`: JPEG, PNG, or WebP, up to 10 MiB by default (the configuration is
  hard-capped at 20 MiB).
- `target`: optional printable text, limited to 64 characters.

Malformed images, MIME mismatches, oversized uploads, and invalid targets are
rejected before inference. The endpoint has a small in-memory per-IP rate
limit suitable for a prototype deployment.

The vision model is not a trained or evaluated bounding-box detector. It maps
ImageNet classification probabilities onto a small indoor-object ontology and
uses approximate Grad-CAM as a localization overlay. Its confidence is raw
and uncalibrated; the default candidate threshold is 0.50 and can be raised
with `PULSEPOINT_RAW_CONFIDENCE_THRESHOLD`. Every response includes metadata
with `status: experimental`, `confidenceCalibration: uncalibrated`,
`validationStatus: unvalidated`,
`localizationMethod: approximate/Grad-CAM`, `assistiveReady: false`, and
`proof: false`. These fields are contractual safety signals: this path must
not be presented as validated assistive sensing.

`GET /health` reports the detector version, status, localization method,
readiness/proof flags, and the supported object ontology. `GET /objects`
returns the ontology labels only.

## Configuration

- `PULSEPOINT_CORS_ORIGINS`: comma-separated explicit `http://` or `https://`
  origins. Wildcards are ignored; an invalid configured value fails closed.
- `PULSEPOINT_MAX_IMAGE_BYTES`: upload limit, bounded to 20 MiB.
- `PULSEPOINT_DETECT_RATE_LIMIT`: detector requests per IP per minute,
  bounded to 1–120 and defaulting to 30.
- `PULSEPOINT_RAW_CONFIDENCE_THRESHOLD`: raw ImageNet probability gate,
  defaulting to 0.50 and never accepted below 0.50.

The default CORS list includes the deployed web origin and the two local Vite
ports. Native mobile requests without an `Origin` header are unaffected by
CORS, but still receive the same request validation and detector metadata.
