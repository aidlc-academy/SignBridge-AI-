# SignBridge AI — Phase 3 Integration Verification Report

**Date:** 2026-10-13
**Scope:** Camera integration + real MediaPipe hand-landmark pipeline (Phase 3)
**Verdict:** ✅ All available automated verification passed. No fabricated predictions. Live webcam behavior requires manual confirmation (steps provided).

---

## 1. Environment

Backend verified against the project's Python 3.11 virtual environment (`backend/.venv311`):

| Package | Version |
|---|---|
| Python | 3.11.9 |
| opencv (cv2) | 4.11.0 |
| mediapipe | 0.10.14 |
| Pillow (PIL) | 10.4.0 |
| numpy | 1.26.4 |
| fastapi | 0.115.2 |

Frontend: Vite 8.x, React + TypeScript, built with `npm run build`.

---

## 2. Features Implemented (Phase 3)

### Backend
- **Real frame-processing endpoint** `POST /api/recognize/landmarks` — accepts a JPEG/PNG frame, runs the actual `VisionService` MediaPipe pipeline, and returns raw landmark coordinates for every detected hand (up to 2).
- **ML inference endpoint** `POST /api/recognize` — decodes the frame, extracts a normalized 63-float feature vector, and runs the classifier. Returns `model_unavailable` honestly when no model is trained.
- **Vision diagnostics** `GET /api/vision/status` — self-check that imports cv2/MediaPipe/Pillow and constructs a MediaPipe `Hands` graph so init failures are visible without sending a frame.
- **Model status** `GET /api/model/status` — reports `loaded: false` until a classifier is trained.
- `VisionService.extract_raw_landmarks()` returns 21 landmarks/hand with handedness; `extract_landmarks()` produces the wrist-normalized 63-float vector.
- The backend never opens `cv2.VideoCapture`; the browser owns capture and sends JPEG frames. Frames are processed in memory and discarded.

### Frontend
- **`useCamera` hook** — `startCamera()` requests `getUserMedia` only on demand; `stopCamera()` stops all tracks and nulls `srcObject`; `captureFrame()` draws the current frame to a canvas and returns a JPEG blob (quality 0.85).
- **`SignRecognition` page** — runs a ~8 fps loop that captures frames and calls the real `/api/recognize/landmarks` endpoint, overlapping-request guarded.
- **Landmark overlay** — `drawLandmarks()` renders 21 points + `HAND_CONNECTIONS` on a canvas, with `object-cover` aspect correction and DPR scaling; fingertips drawn larger.
- **Start / Stop Camera handlers** — wired to the hook with correct disabled-state logic.
- **Dev-only diagnostics panel** — tree-shaken from production builds (`import.meta.env.DEV`).
- **Honest status** — confidence bar is hard-wired to 0 and "Manual Input" is explicitly labeled "demo only". No fabricated gesture output.

---

## 3. Tests Passed

### Backend automated test suite — 25/25 passed (2.24s)
Command: `.venv311\Scripts\python.exe -m pytest tests/ -v`

- `test_api.py` (18 tests): health, model-status fields/messages, gestures vocabulary (10 items), `/api/recognize` input validation (missing/empty/wrong-type), structured recognize response, `/api/recognize/landmarks` validation + black-frame → no_hand + response structure, `/api/vision/status` stack health.
- `test_stabilization.py` (7 tests): sliding-window majority emission, cooldown suppression, reset behavior, none-push handling.

Result: `25 passed, 1 warning` (warning is a benign pytest-asyncio deprecation notice).

### Frontend type check — passed
Command: `npx tsc --noEmit` → exit code 0, no errors.

### Frontend production build — passed
Command: `npm run build` → `tsc -b && vite build`, **2084 modules transformed**, built in ~1.7s.
`dist/assets/SignRecognition-*.js` = 16.06 kB (gzip 5.62 kB). No dev diagnostics code shipped.

---

## 4. Integration Results

Verified through the **actual `VisionService`** and the **actual FastAPI app** (via `fastapi.testclient.TestClient`) using the real hand image `backend/diag_hand.png`.

### Service layer
| Check | Result |
|---|---|
| `self_check()` | `status=ok`, `hands_init=true`, cv2 4.11.0 / mediapipe 0.10.14 / pillow 10.4.0 |
| `process_frame_landmarks(diag_hand.png)` | `status=hand_detected`, **1 hand**, **21 landmarks**, handedness `Left` |
| `process_frame(diag_hand.png)` | `status=ok`, feature vector length **63** |

### HTTP layer (TestClient)
| Endpoint | Result |
|---|---|
| `GET /api/vision/status` | `status=ok` |
| `GET /api/model/status` | `loaded=false` (honest — no model trained yet) |
| `POST /api/recognize/landmarks` | `status=hand_detected`, `hands_detected=1`, **21 landmarks** |
| `POST /api/recognize` | `status=model_unavailable`, `gesture=None`, `confidence=0.0` — **no fake prediction** |

### No fabricated predictions
- Codebase scan for `Math.random`, `fakeGesture`, `mockPredict`, `dummyResult`, hardcoded gestures → **0 matches**.
- `InferenceService.predict()` returns `model_unavailable` with `gesture=None, confidence=0.0` when the model file is absent — it never guesses.
- Frontend `ConfidenceBar` is bound to `0`; recognized-message output is only populated by the explicitly-labeled demo manual-input buttons, not by the recognition loop.

### Camera handlers & resource release (code-verified)
- `stopCamera()` calls `stream.getTracks().forEach(t => t.stop())`, sets `streamRef.current = null`, nulls `videoRef.current.srcObject`, and resets status/dimensions.
- Unmount cleanup `useEffect` stops all tracks, guaranteeing the camera light turns off when leaving the page.
- The recognition-loop `useEffect` clears its `setInterval` and the landmark canvas whenever the camera is not `active`.
- A `track.ended` listener converts an unexpected device disconnect into a visible error state instead of a frozen frame.

---

## 5. Known Limitations

- **No trained classifier yet.** `/api/recognize` returns `model_unavailable` by design until Phase 5 training produces `trained_models/gesture_classifier.joblib`. The landmark endpoint used by Phase 3 does not require a model.
- **Headless hand verification uses a still image.** Automated verification ran MediaPipe against `diag_hand.png` (a real hand photo), not a live webcam, because no camera is available in this environment.
- **Phase 3 displays landmarks only.** The on-screen pipeline shows hand tracking; it does not yet classify or append recognized letters automatically (that is Phase 5/6). The "Recognition Pipeline" card describes the full target flow, not all of which is active yet.
- **Benign startup noise.** MediaPipe/TensorFlow-Lite emit informational stderr lines (XNNPACK delegate, feedback-manager warnings, a protobuf deprecation). These are not errors; all processes exited 0.
- **Secure-context requirement.** `getUserMedia` requires `localhost` or HTTPS; the hook surfaces a clear message otherwise.

---

## 6. Manual Testing Requirements (live webcam)

A real camera could not be exercised in this headless environment. Perform the following manually to confirm end-to-end live behavior.

### Setup
1. Backend: `cd backend` → activate `.venv311` → `uvicorn app.main:app --reload --port 8000`.
2. Frontend: `cd frontend` → `npm run dev` → open `http://localhost:5173`.
3. Navigate to the **Sign to Text** page.

### Test A — Start Camera
- Click **Start Camera**.
- **Expected:** browser prompts for camera permission; on allow, the live mirrored video appears, a green "Camera active" badge shows, and the camera indicator light turns on. Status transitions `idle → requesting → active`.

### Test B — Hand detection + 21-landmark overlay
- Hold an open hand in frame.
- **Expected:** a skeleton overlay of 21 dots + connecting lines tracks your hand; fingertips (indices 4, 8, 12, 16, 20) render as larger amber dots. "Hands detected" shows `1` (or `2` for both hands). Latency displays in ms. Removing your hand clears the overlay and shows "No hand detected".

### Test C — No fake prediction
- **Expected:** the Confidence bar stays at 0% and the "Recognized Message" panel stays empty during live recognition (because no classifier is trained). Status may show "Model not available" — this is correct, not a bug.

### Test D — Stop Camera + resource release
- Click **Stop Camera** (or the top-right stop button).
- **Expected:** video stops, overlay clears, status returns to `idle`, and the **camera hardware light turns off** (confirming tracks were released).

### Test E — Unmount cleanup
- Start the camera, then navigate to another page.
- **Expected:** the camera light turns off immediately on navigation (unmount cleanup stopped the tracks).

### Test F — Permission denied / no device
- Deny the permission prompt (or disconnect the webcam and retry).
- **Expected:** a clear red error banner ("Camera permission was denied…" or "No camera device was found…"); the app does not crash and offers a retry.

---

## 7. Constraints Honored

- UI was not redesigned.
- Phase 4 was not implemented.
- No new dependencies were introduced; verification used existing tooling (`pytest`, `tsc`, `vite`) and the existing `.venv311`.

---

## Summary

Every automated check available in this environment passed: the real MediaPipe pipeline returns **21 landmarks** on a real hand image through both the service and the live FastAPI endpoints, the system emits **no fabricated predictions** (honest `model_unavailable` / `confidence 0.0`), all **25 backend tests pass**, and the **frontend type-checks and builds for production**. Live webcam Start/Stop and resource-release behavior is implemented and code-verified; the manual steps in §6 confirm it on real hardware.
