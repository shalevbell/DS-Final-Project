# Helperviewer

AI-powered interview assistant with real-time video capture and analysis.

## Quick Start

```bash
# Start all services
docker-compose up -d

# Rebuild backend after code changes
docker-compose up -d --build backend

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down
```

## What It Does

- Web-based camera capture interface for interviews
- Real-time video processing backend
- PostgreSQL for data storage
- Redis for caching and session management

## Access

- **Web App**: http://localhost:5555
- **Health Check**: http://localhost:5555/api/health

## Tech Stack

- Frontend: HTML/CSS/JavaScript with camera capture
- Backend: Flask (Python)
- Database: PostgreSQL
- Cache: Redis
- Deployment: Docker Compose

## Project Structure

```
.
├── frontend/          # Web interface with camera capture
├── backend/           # Flask API
│   ├── app.py
│   └── config.py
└── docker-compose.yml
```
