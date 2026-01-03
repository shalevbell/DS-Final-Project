# DS-Final-Project: Helperviewer

AI-Assisted Interview Analysis System - Real-time multimodal interview assistant

## Quick Start

```bash
# Start all services (Flask, PostgreSQL, Redis)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down
```

## API Endpoints

- `GET /` - API info
- `GET /api/health` - Health check (tests Flask, PostgreSQL, and Redis connections)

## Testing

```bash
# Test health endpoint
curl http://localhost:5555/api/health

# Expected response:
# {
#   "status": "healthy",
#   "flask": "running",
#   "services": {
#     "postgresql": {"status": "connected", "message": "Connected"},
#     "redis": {"status": "connected", "message": "Connected"}
#   }
# }
```

## Environment Variables

Environment variables are optional (defaults provided in `backend/config.py`).

To customize, copy `.env.example` to `.env` and modify values:

```bash
cp .env.example .env
```

## Project Structure

```
.
├── backend/
│   ├── app.py              # Flask application
│   ├── config.py           # Configuration with defaults
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile          # Container definition
├── docker-compose.yml      # Service orchestration
└── README.md
```

## Services

- **Flask API**: http://localhost:5555
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

Note: Port 5555 is used instead of 5000 to avoid conflicts with macOS AirPlay Receiver.

---
