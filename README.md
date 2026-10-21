# SignBridge AI

**Every Gesture Deserves a Voice.**

An AI-powered communication assistant that transforms supported sign-language gestures into text and speech, helping people communicate more easily.

---

## Supported Vocabulary

10 ASL static handshapes:

| Sign | Description |
|------|-------------|
| A | Fist with thumb on side |
| B | Flat hand, fingers up, thumb across palm |
| C | Curved hand like holding a cup |
| D | Index up, others curl to touch thumb |
| I | Pinky extended, others curled |
| L | Index up, thumb out — L shape |
| O | All fingers curve to touch thumb |
| V | Index + middle extended (victory/peace) |
| Y | Thumb + pinky extended |
| 1 | Index finger pointing up |

Motion-dependent signs (J, Z) are excluded. ISL support is planned for a future release when suitable training data is available.

---

## Prerequisites

- Node.js 18+ and npm 9+
- Python 3.11+

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The dev server starts at **http://localhost:5173**.

The Vite proxy forwards `/api/*` requests to `http://localhost:8000` during development.

To build for production:

```bash
npm run build
```

---

## Backend Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API starts at **http://localhost:8000**.

Interactive docs: http://localhost:8000/docs

### Environment variables (optional)

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed origins |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/model/status` | Trained model status |
| GET | `/api/gestures` | Supported vocabulary |
| POST | `/api/recognize` | Recognize a JPEG frame |
| GET | `/api/conversations` | List saved conversations |
| POST | `/api/conversations` | Save a conversation |
| DELETE | `/api/conversations/{id}` | Delete a conversation |
| GET | `/api/analytics` | Recognition statistics |

---

## Running Tests

### Backend

```bash
cd backend
pytest tests/ -v
```

### Frontend

TypeScript type-checking:

```bash
cd frontend
npx tsc --noEmit
```

---

## Training the Gesture Classifier

After collecting labeled landmark samples:

```bash
cd backend
python -m app.ml.train_model \
  --dataset datasets/landmarks.csv \
  --output  trained_models/
```

The training script produces:
- `trained_models/gesture_classifier.joblib`
- `trained_models/class_labels.json`
- `trained_models/evaluation_report.json`

Until the model is trained, `GET /api/model/status` returns `loaded: false` and `POST /api/recognize` returns `status: "model_unavailable"` with a clear message.

---

## Project Structure

```
signbridge-ai/
├── frontend/               Vite + React + TypeScript + Tailwind
│   └── src/
│       ├── components/     Reusable UI components
│       ├── pages/          Route-level page components
│       ├── hooks/          useCamera, useSpeech
│       ├── services/       API client
│       ├── store/          Zustand global state
│       └── types/          Shared TypeScript interfaces
│
├── backend/                FastAPI + MediaPipe + scikit-learn
│   ├── app/
│   │   ├── api/            Route handlers
│   │   ├── services/       Vision, inference, stabilization
│   │   ├── models/         SQLAlchemy models + Pydantic schemas
│   │   └── ml/             Training and evaluation pipeline
│   ├── database/           SQLite connection
│   ├── trained_models/     .joblib model files (generated, not committed)
│   ├── datasets/           Landmark CSVs (generated, not committed)
│   └── tests/              pytest test suite
│
└── README.md
```

---

## Privacy

- Camera frames are processed in memory on the backend and discarded immediately.
- No video or audio is stored.
- Conversation history is opt-in and stores text only.
- No biometric landmarks are saved to conversation history.

---

## Disclaimer

SignBridge AI is not a replacement for qualified sign-language interpreters or emergency services (112 / 911).
The current model recognizes only 10 ASL static handshapes and should not be relied upon for critical communication.
