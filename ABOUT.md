# SignBridge AI

### Every Gesture Deserves a Voice.

SignBridge AI is an AI-powered communication assistant that turns sign-language
hand gestures into text and speech — and turns speech back into readable text —
so that signing and non-signing people can hold a conversation without a human
interpreter in the loop.

It is a full-stack web application: a modern React interface that owns the
webcam and the microphone, and a Python computer-vision backend that reads hand
shape from the video in real time. Nothing about the user is stored unless they
explicitly choose to save a conversation, and the system is engineered to **say
"I'm not sure" rather than guess** — it never invents a word it did not
actually recognize.

---

## What the project is

Sign language is a full, natural language expressed through hand shape, motion,
orientation, and facial expression. Most digital tools ignore it entirely: a
deaf or hard-of-hearing signer who walks up to a reception desk, a pharmacy
counter, or a classroom still depends on either a human interpreter being
present or on writing everything down by hand.

SignBridge AI is a step toward closing that gap in software. It watches the
user's hands through an ordinary laptop or phone camera, recognizes the
hand shape, and renders it as on-screen text that can be spoken aloud. In the
other direction, it listens to a hearing person's speech and shows it as text
for the signer to read. The result is a two-way channel that runs entirely in a
web browser against a self-hosted backend.

This is a working application, not a slideshow mock-up. The camera pipeline, the
hand-landmark extraction, the recognition API, the conversation store, the
analytics, the practice trainer, and the emergency screen are all real and
wired together end to end.

---

## What it can do

**See hands and recognize gestures.**
The browser captures webcam frames and streams them to the backend, which runs
Google's MediaPipe Hands to locate 21 skeletal landmarks per hand (knuckles,
finger joints, fingertips, wrist). Those landmarks are normalized and fed to a
gesture classifier that maps the hand shape to a letter or number. A live
skeleton overlay is drawn on top of the video so the user can see exactly what
the system is tracking.

**Translate recognition into text and speech.**
Confident, stabilized recognitions are appended to an on-screen message. The
app can speak that message aloud using the browser's built-in text-to-speech,
giving the signer a literal voice.

**Listen and transcribe speech.**
A speech-to-text screen uses the browser's speech recognition to transcribe a
hearing person's voice into text, so the conversation flows both ways on a
single device.

**Practice and learn.**
A practice mode turns recognition into a guided trainer: the user is prompted to
form a target handshape and gets immediate feedback, which is useful both for
people learning to sign and for calibrating to their own hand and camera.

**Reach help quickly.**
An emergency screen provides fast, pre-written phrases for high-stress
situations, with a clear, prominent reminder that the tool is not a substitute
for professional interpreters or emergency services.

**Remember conversations — only if asked.**
Conversation history is opt-in and stores **text only**. No video, no audio, and
no biometric hand data is ever written to history.

**Show the user how it is doing.**
An analytics dashboard summarizes recognition attempts, acceptance vs.
rejection rates, average latency, per-gesture distribution, and practice
progress — all computed from the local database.

---

## Supported vocabulary

The gesture recognizer focuses on ten static, motion-free ASL handshapes, chosen
because they are distinguishable from a single still frame and form a solid,
honest foundation:

| Sign | Hand shape |
|------|------------|
| A | Fist with the thumb resting along the side |
| B | Flat hand, fingers up, thumb folded across the palm |
| C | Hand curved like it is holding a cup |
| D | Index finger up, the other fingers curl to meet the thumb |
| I | Pinky extended, the rest curled into a fist |
| L | Index up and thumb out, forming an L |
| O | All fingertips curve in to touch the thumb |
| V | Index and middle fingers extended (the "peace"/"victory" shape) |
| Y | Thumb and pinky extended, the middle fingers curled |
| 1 | Single index finger pointing up |

Motion-dependent letters such as **J** and **Z** are intentionally excluded,
because they are defined by movement and cannot be judged from one frame. A
separate, independent word-level module can additionally recognize a small set
of full-word signs ("hello", "thanks", "I love you") from a short sequence of
frames when a compatible sequence model is supplied.

---

## How it works

SignBridge AI is built as two cooperating halves that talk over a small,
well-defined HTTP API.

### 1. The browser owns capture (frontend)

A React + TypeScript single-page app is the only thing that ever touches the
camera and microphone. It:

- requests camera access **on demand**, never automatically;
- draws each frame to a hidden canvas, encodes it as a compact JPEG, and sends
  it to the backend several times per second;
- receives back the detected landmarks and draws the 21-point skeleton overlay;
- appends confident recognitions to the live message and can speak it aloud;
- releases the camera the instant the user stops or navigates away, so the
  hardware indicator light turns off and no stream is left running.

The interface is a polished, accessible, multi-page application with a landing
page, a dashboard, and dedicated screens for sign recognition, speech-to-text,
practice, emergency phrases, history, analytics, and settings. Routes are
code-split so each screen loads only the JavaScript it needs.

### 2. The server reads the hand (backend)

A FastAPI (Python) service receives the frames and does the computer vision.
Crucially, **the server never opens a camera** — it only ever processes the
bytes the browser chooses to send, in memory, and then discards them. For each
frame it:

1. decodes the JPEG into an image;
2. runs **MediaPipe Hands** to find up to two hands and their 21 landmarks each;
3. **normalizes** the landmarks into a scale- and position-independent feature
   vector — the hand is re-centered on the wrist and rescaled by a fixed bone
   length, so the same gesture produces the same features whether the hand is
   near or far, left or right of frame;
4. feeds that feature vector to the trained classifier, which returns a
   probability for each supported sign;
5. returns the most likely sign **only if** the model's confidence clears a
   threshold — otherwise it honestly reports "low confidence" or "no hand".

A separate stabilization layer smooths the stream of per-frame guesses: a sign
is only "confirmed" once it wins a majority vote across a short rolling window,
and it is then briefly suppressed so one steady gesture does not spam the output
with duplicates.

---

## The honesty principle

A recognition system that confidently shows the wrong word is worse than one
that admits uncertainty — especially in a communication aid. SignBridge AI is
built around that idea end to end:

- **No fabricated output.** If no model is loaded, the API returns a clear
  `model_unavailable` status with a helpful message instead of a random guess.
  There is no "demo" code path that makes up letters.
- **Confidence gating.** Predictions below the configured confidence threshold
  are reported as low-confidence and are not appended to the message.
- **Presence gating.** A word is never emitted for a frame in which no hand is
  actually visible, even if a model would otherwise produce a confident answer
  on empty input.
- **Transparent state.** A model-status endpoint tells the frontend exactly
  whether a trained classifier is loaded, what classes it knows, and when it was
  trained, so the UI can reflect reality rather than pretend.

---

## Architecture at a glance

```
Browser (React + TypeScript + Tailwind)
│
│  getUserMedia → canvas → JPEG frame   (several frames / second)
▼
FastAPI backend (Python)
├── Vision service      decode frame → MediaPipe Hands → 21 landmarks
├── Preprocessing       wrist-centered, scale-normalized 63-value feature vector
├── Inference service   trained classifier → predict_proba → confidence-gated label
├── Stabilization       rolling-window majority vote + cooldown
├── Word-sign module    optional sequence model over a 30-frame window (independent)
├── Conversations       opt-in, text-only history (SQLite)
└── Analytics           aggregated recognition statistics (SQLite)
```

### Technology

**Frontend:** React, TypeScript, Vite, Tailwind CSS, React Router, Zustand for
state, the browser Web Speech APIs for text-to-speech and speech-to-text.

**Backend:** FastAPI, Uvicorn, OpenCV and MediaPipe for vision, scikit-learn for
the gesture classifier, NumPy, SQLAlchemy over SQLite for storage, Pydantic for
request/response validation.

---

## The HTTP API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Service health check |
| GET | `/api/vision/status` | Confirms the OpenCV + MediaPipe stack initializes |
| GET | `/api/model/status` | Whether a trained classifier is loaded, and its classes |
| GET | `/api/gestures` | The supported vocabulary and hand-shape descriptions |
| POST | `/api/recognize` | Recognize a single frame → gesture + confidence |
| POST | `/api/recognize/landmarks` | Raw 21-point landmarks for the on-screen overlay |
| GET | `/api/lstm/status` | Whether the optional word-sign model is loaded |
| POST | `/api/recognize/sequence` | Feed one frame into a rolling word-sign window |
| POST | `/api/dataset/sample` | Record one labeled landmark sample for training data |
| GET | `/api/dataset/stats` | Per-label counts of collected training samples |
| GET | `/api/conversations` | List saved (opt-in) conversations |
| POST | `/api/conversations` | Save a conversation (text only) |
| DELETE | `/api/conversations/{id}` | Delete a saved conversation |
| GET | `/api/analytics` | Aggregated recognition and practice statistics |

---

## Privacy and safety

- **Frames are ephemeral.** The backend processes each image in memory and
  discards it immediately. Video is never written to disk.
- **No audio is stored.** Speech is transcribed live in the browser.
- **History is opt-in and text-only.** Saved conversations contain the recognized
  text and nothing else — no video, no audio, no hand landmarks.
- **The camera is user-controlled.** It is requested only on an explicit action
  and released immediately on stop or navigation.
- **Secure context required.** Camera access runs only over `localhost` or HTTPS,
  and the app surfaces a clear message if that requirement is not met.

**This is an assistive tool, not a certified interpreter.** SignBridge AI is not
a replacement for qualified sign-language interpreters or for emergency services.
It should not be relied upon for medical, legal, or life-safety communication.

---

## Why landmarks instead of raw images

SignBridge AI classifies **hand geometry**, not pixels. By reducing each frame to
21 normalized 3D landmarks before recognition, the system becomes largely
independent of skin tone, lighting, background, camera quality, and where in the
frame the hand sits. The same gesture lands on the same feature vector across
very different users and environments. This makes the recognizer smaller, faster,
more robust, and far more private than an image-based model — the raw picture
never has to leave memory, and the stored representation of a gesture is just a
short list of coordinates.

---

## Accessibility

Accessibility is treated as a core requirement, not an afterthought, which is
only fitting for an assistive-technology project: semantic markup, keyboard
navigability, live-region announcements for status changes, visible focus
states, and clear, high-contrast feedback for every recognition state.

---

## Repository layout

```
signbridge-ai/
├── frontend/                 React + TypeScript + Tailwind single-page app
│   └── src/
│       ├── pages/            Landing, Dashboard, Sign, Speech, Practice,
│       │                     Emergency, History, Analytics, Settings
│       ├── components/       Layout shell + reusable UI primitives
│       ├── hooks/            Camera and speech hooks
│       ├── services/         Typed API client
│       ├── store/            Global app state
│       └── types/            Shared TypeScript interfaces
│
├── backend/                  FastAPI + MediaPipe + scikit-learn service
│   ├── app/
│   │   ├── api/              Route handlers (recognition, dataset,
│   │   │                     conversations, analytics)
│   │   ├── services/         Vision, inference, stabilization, word-sign
│   │   ├── models/           Database models + request/response schemas
│   │   └── ml/               Preprocessing + classifier training pipeline
│   ├── database/             SQLite connection and session management
│   ├── trained_models/       Classifier artifacts (generated, not committed)
│   ├── datasets/             Collected landmark data (generated, not committed)
│   └── tests/                Automated test suite
│
└── README.md                 Setup, run, and development guide
```

---

## Project status and direction

The platform — camera pipeline, landmark extraction, recognition API,
stabilization, conversation store, analytics, practice trainer, and the full
multi-page interface — is implemented and working. The static-handshape
recognizer covers the ten signs above, with an independent word-level module
available for short motion signs.

Natural next directions include widening the vocabulary toward the full ASL
alphabet and common words, adding support for additional sign languages (such as
ISL) once suitable training data is available, and incorporating facial and
upper-body cues that carry grammatical meaning in sign language.
