"""Lightweight contract tests; deliberately independent of torch/Pillow."""

import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    from .contract import (
        DEFAULT_CORS_ORIGINS,
        DETECTOR_VERSION,
        detector_metadata,
        cors_origins,
        validate_image_headers,
        validate_target,
    )
except ImportError:  # Supports `cd server && python -m unittest test_contract.py`.
    from contract import (
        DEFAULT_CORS_ORIGINS,
        DETECTOR_VERSION,
        detector_metadata,
        cors_origins,
        validate_image_headers,
        validate_target,
    )


ROOT = Path(__file__).resolve().parent


class BoundaryContractTests(unittest.TestCase):
    def test_metadata_can_never_claim_assistive_readiness(self):
        metadata = detector_metadata()
        self.assertEqual(metadata["modelVersion"], DETECTOR_VERSION)
        self.assertEqual(metadata["detectorVersion"], DETECTOR_VERSION)
        self.assertEqual(metadata["status"], "experimental")
        self.assertEqual(metadata["validationStatus"], "unvalidated")
        self.assertEqual(metadata["confidenceCalibration"], "uncalibrated")
        self.assertEqual(metadata["localizationMethod"], "approximate/Grad-CAM")
        self.assertFalse(metadata["assistiveReady"])
        self.assertFalse(metadata["proof"])
        self.assertFalse(metadata["assistiveReadyProof"])

    def test_cors_is_explicit_and_wildcards_fail_closed(self):
        self.assertNotIn("*", DEFAULT_CORS_ORIGINS)
        with patch.dict(os.environ, {"PULSEPOINT_CORS_ORIGINS": "https://app.example, *, bad"}):
            self.assertEqual(cors_origins(), ["https://app.example"])

    def test_target_and_image_limits(self):
        self.assertEqual(validate_target("  mug  "), "mug")
        with self.assertRaises(ValueError):
            validate_target("x" * 65)
        validate_image_headers("image/jpeg", 128)
        with self.assertRaises(ValueError):
            validate_image_headers("application/octet-stream", 128)
        with self.assertRaises(OverflowError):
            validate_image_headers("image/jpeg", 11 * 1024 * 1024)

    def test_server_source_has_legacy_and_versioned_detection_routes(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        routes = {
            decorator.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "post"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
        }
        self.assertIn("/detect", routes)
        self.assertIn("/v1/detect", routes)

    def test_model_source_does_not_inflate_confidence_or_claim_readiness(self):
        source = (ROOT / "model.py").read_text(encoding="utf-8")
        self.assertNotIn("best_conf * 3.5", source)
        self.assertIn("'confidence': best_conf", source)
        self.assertIn("'assistiveReady': False", source)
        self.assertIn("'proof': False", source)


if __name__ == "__main__":
    unittest.main()
