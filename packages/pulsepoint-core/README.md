# Pulse Point core

This package contains platform-neutral target, detection, tracking, guidance,
and state-machine logic. It intentionally has no React, camera, Expo, browser,
network, or filesystem dependencies.

Detections use normalized coordinates and carry source and distance provenance.
The `assistiveReady` flag is conservative: simulated/offline detections never
become assistive-ready, and callers must opt in for other experimental sources.
