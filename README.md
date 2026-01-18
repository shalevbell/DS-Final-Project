# Helperviewer

AI-powered interview assistant with real-time video analysis.

## Quick Start

```bash
# Start all services
docker-compose up -d

# Rebuild after code changes
docker-compose up -d --build

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## What It Does

**Real-time Interview Analysis:**
- Captures video/audio in 30-second chunks via WebRTC
- Processes chunks in parallel with ML models:
  - **Whisper**: Speech transcription
  - **MediaPipe**: Facial expressions, gestures, posture
  - **Vocal Tone**: Pitch, tempo, emotional analysis
- Stores results in Redis for real-time feedback

## Access

- **Web App**: http://localhost:5555
- **Health Check**: http://localhost:5555/api/health
- **Processing Stats**: http://localhost:5555/api/health/processing

## Architecture

```
Frontend (WebRTC)
  → 30s chunks via WebSocket
    → Backend (FFmpeg → MP4/WAV)
      → Redis PUBSUB
        → Chunk Processor (parallel ML analysis)
          → Results stored in Redis
```

## Tech Stack

- **Frontend**: HTML/CSS/JavaScript with WebRTC camera capture
- **Backend**: Flask + SocketIO + eventlet
- **Video Processing**: FFmpeg (MP4/WAV conversion)
- **ML Models**: Whisper, MediaPipe, Vocal Tone (placeholders for now)
- **Database**: PostgreSQL
- **Cache/Queue**: Redis (with PUBSUB)
- **Deployment**: Docker Compose

## Project Structure

```
.
├── frontend/              # Web interface
│   ├── index.html         # Main UI
│   ├── css/               # Styles
│   └── js/
│       └── app.js         # WebRTC + WebSocket logic
│
├── backend/               # Flask application
│   ├── app.py             # Main entry point
│   ├── config.py          # Configuration
│   │
│   ├── routes/            # HTTP & WebSocket handlers
│   │   ├── http_routes.py          # Health checks, frontend serving
│   │   └── websocket_handlers.py   # SocketIO events
│   │
│   ├── services/          # Business logic
│   │   ├── connection_manager.py   # Redis/Postgres connections
│   │   ├── video_processor.py      # FFmpeg processing
│   │   └── chunk_storage.py        # Redis storage operations
│   │
│   ├── chunk_processor.py # Background worker (PUBSUB subscriber)
│   ├── processing_queue.py # Thread-safe queue
│   ├── run_models.py      # ML model registry & parallel execution
│   │
│   └── utils/
│       └── redis_helper.py # Redis key formatting utilities
│
│
└── docker-compose.yml     # Service orchestration
```

## How Chunking Works

1. **Frontend** records 30-second video/audio chunks
2. **WebSocket** sends chunks to backend
3. **FFmpeg** converts to MP4 (video) + WAV (audio)
4. **Redis** stores chunks and publishes PUBSUB notification
5. **ChunkProcessor** picks up notification from queue
6. **ML Models** run in parallel (ThreadPoolExecutor):
   - Whisper transcribes audio
   - MediaPipe analyzes video (facial expressions, gestures)
   - VocalTone analyzes audio (pitch, tempo, emotion)
7. **Results** stored back in Redis for retrieval

## Development

**Adding a new ML model:**

1. Implement analysis function in `backend/run_models.py`:
   ```python
   def analyze_new_model(bytes, session_id, chunk_index) -> Dict:
       # Your model logic here
       return {'result': 'data', 'processing_time_ms': 100}
   ```

2. Register in `MODEL_REGISTRY`:
   ```python
   MODEL_REGISTRY = {
       'new_model': {
           'function': analyze_new_model,
           'data_type': 'audio'  # or 'video'
       }
   }
   ```

That's it! The system will automatically:
- Process your new model for all chunks
- Run it in parallel with other models (auto-scales worker count)

## Environment Variables

See `backend/config.py` for available options:
- `REDIS_URL` - Redis connection string
- `DATABASE_URL` - PostgreSQL connection string
- `PROCESSING_MAX_WORKERS` - Parallel chunk processing workers (default: 3)
- `CHUNK_DURATION_MS` - Chunk duration in milliseconds (default: 30000)
