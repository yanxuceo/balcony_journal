# ADR-001: Image resolution and capture trigger strategy

**Date**: 2026-04-19  
**Status**: Accepted  
**Context**: Plant symbiote bird detection pipeline

## Decision

1. **ESP32-CAM outputs SVGA 800×600** for both gating and VLM input — single resolution throughout the pipeline.
2. **Capture is motion-driven** on the ESP32 side (frame differencing), not Jetson-triggered.
3. **Fixed 15-minute interval captures** also flow through, for timeline and non-event archival.
4. **Jetson never sends "please re-capture at higher resolution" commands to ESP32.**

## Reasoning

### Why not two-stage (low-res gating → trigger high-res)?

The naive design is: ESP32 sends low-res continuously → YOLO gates → on positive,
Jetson asks ESP32 to re-capture in UXGA → send to Gemma.

This fails because:

| Step                                  | Latency      |
| ------------------------------------- | ------------ |
| Jetson → ESP32 command round-trip     | ~100 ms      |
| ESP32 framesize reconfigure           | 200–500 ms   |
| UXGA capture + JPEG encode            | 500–1000 ms  |
| ESP32 → Jetson image transfer         | 500–1000 ms  |
| **Total trigger-to-image latency**    | **1.3–2.6 s**|

Typical bird dwell time on a balcony is **2–15 seconds**. The trigger
delay is comparable to or exceeds the event duration — by the time
the high-res image arrives, the bird has likely flown off.

### Why SVGA 800×600 is enough for Gemma

From Gemma 4 E2B startup logs: `image_min_pixels = 580608`, roughly
762×762. The vision encoder internally resizes all inputs to this
range. Giving Gemma UXGA (1600×1200) provides no additional signal —
it downsamples to ~800×800 internally anyway.

**Counterintuitive but important**: vision model token budgets are
fixed; more pixels in don't mean more detail out.

### Why motion-driven (not periodic) capture

- 99%+ of the time, the balcony has no event worth analyzing.
- Continuously streaming SVGA at 1 FPS costs ~2 Mbps and burns GPU
  doing YOLO on empty frames.
- ESP32 frame differencing is cheap (tens of ms), catches bird
  arrival within one frame, and drops everything else before it
  leaves the device.
- YOLO gating on Jetson then filters the ~50–200 motion events per
  day down to real bird events (the rest is leaves, light changes,
  shadows, etc.).

### Why single-direction upload (ESP32 → Jetson only)

Bidirectional control (Jetson commanding ESP32) is:
- Slower (round-trip latency)
- Less reliable (more failure modes)
- Harder to debug (distributed state)

Push-only from ESP32 keeps the edge node simple and the orchestration
one-way.

## Consequences

- ESP32 firmware must implement: SVGA capture, frame differencing,
  HTTP POST upload, 15-minute cron loop.
- Jetson gating accepts a single resolution — no adaptive preprocessing.
- Archival images are SVGA; "zoom in on this bird's feathers" is not
  supported. If higher detail is ever needed, it requires a hardware
  change (different camera module), not a protocol change.
- Bandwidth: ~20 MB/day at typical event rates, negligible.

## Alternatives considered

- **Always UXGA**: 3–4× bandwidth, 3–4× storage, no visible quality
  gain through Gemma's encoder. Rejected.
- **Jetson-triggered high-res capture**: fails on latency (see above).
  Rejected.
- **Continuous video stream + on-device inference**: ESP32-S3 lacks
  the compute for reliable real-time YOLO. Would require a Pi or
  larger MCU at the edge. Rejected on cost/complexity.
