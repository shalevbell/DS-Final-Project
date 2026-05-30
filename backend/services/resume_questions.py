"""
Resume-based warmup question generator.

Extracts text from a PDF resume and uses a lightweight Ollama model to generate
5 general interview starter questions. Questions are streamed one at a time every
6 seconds to fill the gap before the first video chunk completes processing.
"""

import base64
import logging
import re
import io

import eventlet
import requests

from config import Config
from services.text_streaming import stream_text

logger = logging.getLogger(__name__)

_MAX_RESUME_CHARS = 3000
_QUESTIONS_TO_GENERATE = 5
_STREAM_INTERVAL_SECONDS = 2


def extract_pdf_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    full_text = "\n".join(pages_text).strip()
    return full_text[:_MAX_RESUME_CHARS]


def stream_resume_questions(session_id: str, pdf_bytes: bytes, target_role: str, socketio) -> None:
    """
    Generate and stream resume-based warmup questions.

    Intended to be spawned as an eventlet greenthread immediately when a session
    starts so that questions fill the gap before the first chunk finishes processing.
    """
    try:
        resume_text = extract_pdf_text(pdf_bytes)
        if not resume_text:
            logger.warning("[ResumeQuestions] No text extracted from resume PDF, skipping")
            return

        role_context = f" for a {target_role} position" if target_role else ""
        user_prompt = (
            f"Here is a candidate's resume:\n\n{resume_text}\n\n"
            f"Generate exactly {_QUESTIONS_TO_GENERATE} interview questions{role_context}. "
            f"Mix the following types across the {_QUESTIONS_TO_GENERATE} questions:\n"
            "- Questions anchored to a specific project or achievement mentioned in the resume\n"
            "- Questions about a specific employer — what drew them there, what they learned, or what the culture/team was like\n"
            "- At least one question about why they are looking for a new opportunity or what they are seeking next in their career\n"
            "Do not ask generic questions that could apply to any candidate. "
            "Keep each question to one concise sentence. "
            f"Output only the {_QUESTIONS_TO_GENERATE} questions, numbered 1 through {_QUESTIONS_TO_GENERATE}, one per line. No preamble."
        )

        base_url = Config.OLLAMA_BASE_URL.rstrip("/")
        model_name = Config.OLLAMA_RESUME_MODEL_NAME

        logger.info(f"[ResumeQuestions] Generating warmup questions for session {session_id} using {model_name}")

        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an experienced interviewer. Generate concise, thoughtful interview questions."
                    },
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.5, "num_predict": 400},
            },
            timeout=Config.OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        raw = response.json().get("message", {}).get("content", "")

        questions = _parse_questions(raw)
        if not questions:
            logger.warning(f"[ResumeQuestions] Could not parse questions from model response for session {session_id}")
            return

        logger.info(f"[ResumeQuestions] Streaming {len(questions)} questions for session {session_id}")

        for i, question in enumerate(questions, start=1):
            stream_text(
                socketio,
                question,
                session_id,
                {"source": "resume_questions", "chunk": i},
            )
            if i < len(questions):
                eventlet.sleep(_STREAM_INTERVAL_SECONDS)

        logger.info(f"[ResumeQuestions] Done streaming warmup questions for session {session_id}")

    except Exception as e:
        logger.warning(f"[ResumeQuestions] Failed to generate warmup questions for session {session_id}: {e}")


def _parse_questions(text: str) -> list:
    questions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1.", "1)", "- "
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        line = line.strip()
        if line and "?" in line:
            questions.append(line)
        if len(questions) == _QUESTIONS_TO_GENERATE:
            break
    return questions


def decode_resume_data_url(data_url: str) -> bytes:
    """Convert a base64 data URL (data:application/pdf;base64,...) to raw bytes."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)
