"""
Build and persist a professional interview session conclusion after stream ends.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional

import eventlet

from services.connection_manager import get_redis_client
from services.db_service import get_session_with_chunks, save_session_conclusion, get_session_conclusion
from services.text_streaming import queue_socket_emit

logger = logging.getLogger(__name__)

_QUESTION_SOURCES = {'resume_questions', 'interviewer_ollama'}
_CONCLUSION_WAIT_SECONDS = 18
_CONCLUSION_POLL_INTERVAL = 3
_CONCLUSION_MAX_POLLS = 8


def record_streamed_question(session_id: Optional[str], source: str, chunk: Any, text: str) -> None:
    """Append a streamed question to Redis for later conclusion assembly."""
    if not session_id or source not in _QUESTION_SOURCES or not text:
        return

    r = get_redis_client()
    if not r:
        return

    questions = _extract_questions_from_text(text)
    if not questions:
        return

    key = f'session:{session_id}:questions'
    try:
        for question in questions:
            entry = json.dumps({
                'source': source,
                'chunk': chunk,
                'text': question,
            })
            r.rpush(key, entry)
        r.expire(key, 7200)
    except Exception as e:
        logger.warning(f'[Conclusion] Failed to record question for {session_id}: {e}')


def _extract_questions_from_text(text: str) -> List[str]:
    questions = []
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line).strip()
        if line and '?' in line:
            questions.append(line)
    if not questions and text and '?' in text:
        questions.append(text.strip())
    return questions


def _load_recorded_questions(session_id: str) -> List[Dict]:
    r = get_redis_client()
    if not r:
        return []
    try:
        raw_items = r.lrange(f'session:{session_id}:questions', 0, -1)
        items = []
        for raw in raw_items or []:
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8')
            items.append(json.loads(raw))
        return items
    except Exception as e:
        logger.warning(f'[Conclusion] Failed to load recorded questions for {session_id}: {e}')
        return []


def _normalize_results(model_results) -> Dict:
    if isinstance(model_results, str):
        try:
            return json.loads(model_results)
        except json.JSONDecodeError:
            return {}
    return model_results or {}


def _questions_from_chunks(chunks: List[Dict]) -> List[Dict]:
    collected = []
    for chunk in chunks:
        chunk_index = chunk.get('chunk_index')
        model_results = _normalize_results(chunk.get('model_results'))
        ollama = model_results.get('interviewer_ollama') or {}
        if isinstance(ollama, dict) and ollama.get('questions'):
            for q in ollama['questions']:
                if q and '?' in q:
                    collected.append({
                        'source': 'interviewer_ollama',
                        'chunk': chunk_index,
                        'text': q.strip(),
                    })
    return collected


def _merge_questions(recorded: List[Dict], from_chunks: List[Dict]) -> Dict[str, List[Dict]]:
    seen = set()
    resume_warmup = []
    ai_followup = []

    for item in recorded + from_chunks:
        text = (item.get('text') or '').strip()
        if not text or text in seen:
            continue
        seen.add(text)
        source = item.get('source', 'unknown')
        entry = {
            'text': text,
            'chunk': item.get('chunk'),
            'source': source,
        }
        if source == 'resume_questions':
            resume_warmup.append(entry)
        else:
            ai_followup.append(entry)

    return {
        'resume_warmup': resume_warmup,
        'ai_followup': ai_followup,
        'total_count': len(resume_warmup) + len(ai_followup),
    }


def _aggregate_analytics(chunks: List[Dict]) -> Dict[str, Any]:
    stress_levels = []
    engagement_scores = []
    posture_scores = []
    emotions = []
    voice_emotions = []
    clifton_domains = []
    dev_opportunities = []
    eye_contact_yes = 0
    eye_contact_total = 0
    body_stable_yes = 0
    body_stable_total = 0

    for chunk in chunks:
        results = _normalize_results(chunk.get('model_results'))
        mp = results.get('mediapipe') or {}
        vt = results.get('vocaltone') or {}
        cf = results.get('clifton_fusion') or {}

        if isinstance(mp, dict) and 'error' not in mp:
            if isinstance(mp.get('stress_level'), (int, float)):
                stress_levels.append(float(mp['stress_level']))
            if isinstance(mp.get('engagement_score'), (int, float)):
                engagement_scores.append(float(mp['engagement_score']))
            if isinstance(mp.get('posture_score'), (int, float)):
                posture_scores.append(float(mp['posture_score']))
            if mp.get('dominant_emotion'):
                emotions.append(str(mp['dominant_emotion']).lower())
            if isinstance(mp.get('eye_contact'), bool):
                eye_contact_total += 1
                if mp['eye_contact']:
                    eye_contact_yes += 1
            if isinstance(mp.get('body_stable'), bool):
                body_stable_total += 1
                if mp['body_stable']:
                    body_stable_yes += 1

        if isinstance(vt, dict) and 'error' not in vt and vt.get('emotion'):
            voice_emotions.append(str(vt['emotion']).lower())

        if isinstance(cf, dict) and 'error' not in cf:
            if cf.get('predicted_domain'):
                clifton_domains.append(str(cf['predicted_domain']))
            for area in cf.get('development_opportunities') or []:
                if area and area not in dev_opportunities:
                    dev_opportunities.append(area)

    emotion_counter = Counter(emotions)
    voice_counter = Counter(voice_emotions)
    domain_counter = Counter(clifton_domains)

    return {
        'chunks_analyzed': len(chunks),
        'avg_stress_level': round(mean(stress_levels), 1) if stress_levels else None,
        'avg_engagement_pct': round(mean(engagement_scores) * 100, 1) if engagement_scores else None,
        'avg_posture_pct': round(mean(posture_scores) * 100, 1) if posture_scores else None,
        'dominant_emotions': [e for e, _ in emotion_counter.most_common(3)],
        'dominant_voice_emotions': [e for e, _ in voice_counter.most_common(3)],
        'primary_clifton_domain': domain_counter.most_common(1)[0][0] if domain_counter else None,
        'development_areas': dev_opportunities[:5],
        'eye_contact_consistency_pct': round((eye_contact_yes / eye_contact_total) * 100, 1) if eye_contact_total else None,
        'posture_stability_pct': round((body_stable_yes / body_stable_total) * 100, 1) if body_stable_total else None,
    }


def _format_duration(started_at, ended_at) -> Optional[str]:
    if not started_at or not ended_at:
        return None
    if isinstance(started_at, str):
        started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
    if isinstance(ended_at, str):
        ended_at = datetime.fromisoformat(ended_at.replace('Z', '+00:00'))
    delta = ended_at - started_at
    minutes = int(delta.total_seconds() // 60)
    seconds = int(delta.total_seconds() % 60)
    if minutes <= 0:
        return f'{seconds}s'
    return f'{minutes}m {seconds}s'


def _build_highlights(analytics: Dict[str, Any], question_groups: Dict[str, Any]) -> List[str]:
    highlights = []
    if question_groups.get('total_count'):
        highlights.append(f"{question_groups['total_count']} interview questions were generated across the session.")
    if analytics.get('avg_engagement_pct') is not None:
        highlights.append(f"Average engagement remained around {analytics['avg_engagement_pct']}%.")
    if analytics.get('eye_contact_consistency_pct') is not None:
        highlights.append(f"Eye contact was maintained in roughly {analytics['eye_contact_consistency_pct']}% of analyzed segments.")
    if analytics.get('primary_clifton_domain'):
        highlights.append(f"Clifton analysis most often indicated the {analytics['primary_clifton_domain']} domain.")
    if analytics.get('dominant_voice_emotions'):
        highlights.append(
            f"Voice tone trends included: {', '.join(analytics['dominant_voice_emotions'][:2])}."
        )
    return highlights[:5]


def _build_recommendations(analytics: Dict[str, Any], question_groups: Dict[str, Any]) -> List[str]:
    recs = []
    if analytics.get('development_areas'):
        recs.append(
            'Follow up on development areas: ' + ', '.join(analytics['development_areas'][:3]) + '.'
        )
    if analytics.get('avg_stress_level') is not None and analytics['avg_stress_level'] >= 60:
        recs.append('Consider a second conversation focused on situational comfort and team fit under pressure.')
    if question_groups.get('ai_followup'):
        recs.append('Review AI follow-up questions and note which topics produced the strongest candidate responses.')
    if not recs:
        recs.append('Schedule a structured debrief with hiring stakeholders while session insights are still fresh.')
    return recs[:4]


def _build_executive_summary(session: Dict, analytics: Dict, question_groups: Dict) -> str:
    name = session.get('candidate_name') or 'The candidate'
    role = session.get('target_role')
    duration = _format_duration(session.get('started_at'), session.get('ended_at'))
    parts = [f'Interview session with {name} has concluded.']
    if role:
        parts.append(f'The evaluation targeted the {role} role.')
    if duration:
        parts.append(f'Session duration was approximately {duration}.')
    parts.append(
        f"A total of {question_groups.get('total_count', 0)} questions were presented, "
        f"including resume-based warm-up and AI-generated follow-ups."
    )
    if analytics.get('avg_stress_level') is not None:
        parts.append(f"Observed stress level averaged {analytics['avg_stress_level']}%.")
    if analytics.get('primary_clifton_domain'):
        parts.append(f"Strengths signal leaned toward {analytics['primary_clifton_domain']}.")
    return ' '.join(parts)


def build_session_conclusion(session_id: str) -> Optional[Dict[str, Any]]:
    """Assemble a full session conclusion from DB, Redis, and chunk analytics."""
    data = get_session_with_chunks(session_id)
    if not data:
        return None

    session = data['session']
    chunks = data['chunks'] or []
    recorded = _load_recorded_questions(session_id)
    from_chunks = _questions_from_chunks(chunks)
    question_groups = _merge_questions(recorded, from_chunks)
    analytics = _aggregate_analytics(chunks)

    resume_info = None
    if session.get('resume_filename'):
        resume_info = {
            'filename': session['resume_filename'].split('_', 1)[-1],
            'stored_filename': session['resume_filename'],
            'download_url': f'/api/sessions/{session_id}/resume',
        }

    conclusion = {
        'session_id': session_id,
        'candidate_name': session.get('candidate_name'),
        'target_role': session.get('target_role'),
        'interview_requirements': session.get('interview_requirements'),
        'started_at': session.get('started_at').isoformat() if session.get('started_at') else None,
        'ended_at': session.get('ended_at').isoformat() if session.get('ended_at') else None,
        'duration': _format_duration(session.get('started_at'), session.get('ended_at')),
        'chunks_processed': len(chunks),
        'resume': resume_info,
        'questions': question_groups,
        'analytics_summary': analytics,
        'highlights': _build_highlights(analytics, question_groups),
        'recommended_next_steps': _build_recommendations(analytics, question_groups),
        'executive_summary': _build_executive_summary(session, analytics, question_groups),
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    return conclusion


def generate_and_emit_conclusion(session_id: str, socketio) -> None:
    """
    Wait briefly for final chunk processing, build conclusion, persist, and emit.
    Runs in an eventlet greenthread. Blocking DB work uses tpool; emits use the
    thread-safe queue so we never call socketio.emit from the wrong thread.
    """
    del socketio  # emits go through queue_socket_emit on the main hub worker

    try:
        existing = eventlet.tpool.execute(get_session_conclusion, session_id)
        if existing and existing.get('conclusion'):
            queue_socket_emit('session_conclusion', {
                'sessionId': session_id,
                'conclusion': existing['conclusion'],
            }, broadcast=True)
            return

        eventlet.sleep(_CONCLUSION_WAIT_SECONDS)

        conclusion = None
        for _attempt in range(_CONCLUSION_MAX_POLLS):
            conclusion = eventlet.tpool.execute(build_session_conclusion, session_id)
            if conclusion and (
                conclusion.get('chunks_processed', 0) > 0
                or conclusion['questions'].get('total_count', 0) > 0
            ):
                break
            eventlet.sleep(_CONCLUSION_POLL_INTERVAL)

        if not conclusion:
            logger.warning(f'[Conclusion] Could not build conclusion for {session_id}')
            return

        eventlet.tpool.execute(save_session_conclusion, session_id, conclusion)

        queue_socket_emit('session_conclusion', {
            'sessionId': session_id,
            'conclusion': conclusion,
        }, broadcast=True)
        logger.info(f'[Conclusion] Generated and emitted for session {session_id}')

    except Exception as e:
        logger.error(f'[Conclusion] Failed for session {session_id}: {e}', exc_info=True)
