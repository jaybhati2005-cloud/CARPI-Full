import os
import asyncio
import random
import tempfile
import json
import re
import time
from typing import List

import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, Field

from backend.audio_analysis import analyze_audio


load_dotenv()

# ---------------- Multi-provider AI configuration ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Keep model IDs configurable: provider model availability and free-tier limits change.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODELS = [
    item.strip() for item in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    if item.strip() and item.strip() != MODEL
]
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
GROQ_AUDIO_MODEL = os.getenv("GROQ_AUDIO_MODEL", "whisper-large-v3-turbo").strip()
# Off by default to honor the free-only setup; enable only after checking your Groq billing.
GROQ_AUDIO_FALLBACK = os.getenv("GROQ_AUDIO_FALLBACK", "false").strip().lower() in {"1", "true", "yes", "on"}
AI_PROVIDER_ORDER = [
    item.strip().lower()
    for item in os.getenv("AI_PROVIDER_ORDER", "gemini,groq").split(",")
    if item.strip().lower() in {"gemini", "groq"}
]
MAX_ATTEMPTS_PER_MODEL = max(1, min(4, int(os.getenv("AI_MAX_ATTEMPTS", os.getenv("GEMINI_MAX_ATTEMPTS", "2")))))
AI_REQUEST_TIMEOUT = max(10, int(os.getenv("AI_REQUEST_TIMEOUT", "75")))

# Import-safe: the app starts even if one provider key is absent.
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
# Retain the old variable name for any local code that may import it.
client = gemini_client

import httpx
_groq_http = httpx.Client(
    base_url="https://api.groq.com/openai/v1/",
    timeout=httpx.Timeout(AI_REQUEST_TIMEOUT, connect=10.0),
) if GROQ_API_KEY else None

app = FastAPI(title="CareerPilot AI v2 - Multi Provider")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Structured Gemini response schemas ----------
class JobAnalysis(BaseModel):
    job_role: str
    match_score: int = Field(ge=0, le=100)
    candidate_summary: str
    strong_matches: List[str]
    partial_matches: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    reasoning: List[str]

class ApplicationMaterials(BaseModel):
    application_description: str
    why_this_role: str
    cover_letter: str
    key_points_to_highlight: List[str]

class RoleRecommendation(BaseModel):
    role: str
    fit_score: int = Field(ge=0, le=100)
    why: str
    skills_to_improve: List[str]

class RoleRecommendations(BaseModel):
    roles: List[RoleRecommendation]

class InterviewQuestion(BaseModel):
    question: str
    category: str
    purpose: str

class InterviewPlan(BaseModel):
    role: str
    questions: List[InterviewQuestion]

class InterviewEvaluation(BaseModel):
    relevance: int = Field(ge=0, le=10)
    technical_correctness: int = Field(ge=0, le=10)
    completeness: int = Field(ge=0, le=10)
    structure: int = Field(ge=0, le=10)
    clarity: int = Field(ge=0, le=10)
    overall_score: int = Field(ge=0, le=100)
    strengths: List[str]
    improvements: List[str]
    better_answer_direction: str

class Followup(BaseModel):
    follow_up_question: str
    reason: str

class TranscriptResult(BaseModel):
    transcript: str

class FinalInterviewReport(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    content_score: int = Field(ge=0, le=100)
    communication_score: int = Field(ge=0, le=100)
    delivery_score: int = Field(ge=0, le=100)
    summary: str
    strengths: List[str]
    improvements: List[str]
    recurring_issues: List[str]
    personalized_plan: List[str]
    observable_behavior_summary: str


def extract_pdf_text(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if not text.strip():
            raise HTTPException(400, "The PDF contains no extractable text.")
        return text.strip()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read PDF: {e}")


def _error_text(exc: Exception) -> str:
    return str(exc).lower()


def _is_retryable_error(exc: Exception) -> bool:
    """Return True for transient capacity, rate-limit and network failures."""
    message = _error_text(exc)
    return any(token in message for token in (
        "429", "resource_exhausted", "too many requests", "rate limit",
        "quota exceeded", "503", "502", "500", "504", "service unavailable",
        "temporarily unavailable", "overloaded", "deadline exceeded",
        "timed out", "timeout", "connection reset", "connection error",
    ))


def _backoff(attempt: int) -> None:
    # Bounded exponential backoff with jitter. This runs in a worker thread.
    delay = min(8.0, 1.5 * (2 ** attempt)) + random.uniform(0.1, 0.8)
    time.sleep(delay)


def _gemini_generate(contents, schema: type[BaseModel] | None = None):
    if not gemini_client:
        raise RuntimeError("Gemini is not configured (GEMINI_API_KEY is missing).")

    models = [MODEL, *GEMINI_FALLBACK_MODELS]
    last_error = None
    for model_name in models:
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                config = None
                if schema is not None:
                    config = {
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    }
                return gemini_client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < MAX_ATTEMPTS_PER_MODEL and _is_retryable_error(exc):
                    print(f"[AI] Gemini {model_name} transient failure; retry {attempt + 1}.")
                    _backoff(attempt)
                    continue
                # Move to the next Gemini model/provider for model, auth, or quota errors.
                print(f"[AI] Gemini model {model_name} failed: {str(exc)[:220]}")
                break
    raise RuntimeError(f"Gemini failed: {last_error}") from last_error


def _groq_generate(prompt: str, schema: type[BaseModel]) -> str:
    """Call Groq's OpenAI-compatible Chat Completions API in JSON mode."""
    if not GROQ_API_KEY or not _groq_http:
        raise RuntimeError("Groq is not configured (GROQ_API_KEY is missing).")

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    system_prompt = (
        "You are CareerPilot, an accurate career and interview assistant. "
        "Return exactly one valid JSON object matching this JSON Schema. "
        "Do not wrap it in Markdown fences and do not add commentary.\n"
        f"JSON Schema: {schema_json}"
    )
    last_error = None
    for attempt in range(MAX_ATTEMPTS_PER_MODEL):
        try:
            response = _groq_http.post(
                "chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_ATTEMPTS_PER_MODEL and _is_retryable_error(exc):
                print(f"[AI] Groq transient failure; retry {attempt + 1}.")
                _backoff(attempt)
                continue
            break
    raise RuntimeError(f"Groq failed: {last_error}") from last_error


def _available_providers() -> list[str]:
    configured = {"gemini": bool(gemini_client), "groq": bool(GROQ_API_KEY and _groq_http)}
    return [name for name in AI_PROVIDER_ORDER if configured.get(name)]


def _parse_and_validate(raw_text: str, schema: type[BaseModel]) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw_text or "").strip(), flags=re.I)
    parsed = json.loads(cleaned)
    return schema.model_validate(parsed).model_dump()


def ask_gemini(prompt: str, schema: type[BaseModel]) -> dict:
    """Backward-compatible function name; routes through Gemini then Groq."""
    providers = _available_providers()
    if not providers:
        raise HTTPException(
            503,
            "No AI provider is configured. Add GEMINI_API_KEY and/or GROQ_API_KEY to .env.",
        )

    failures: list[str] = []
    for provider in providers:
        try:
            if provider == "gemini":
                response = _gemini_generate(
                    prompt,
                    schema=schema,
                )
                raw_text = response.text or ""
            else:
                raw_text = _groq_generate(prompt, schema)

            # Validate against the exact existing response schema before returning.
            return _parse_and_validate(raw_text, schema)
        except Exception as exc:
            failures.append(f"{provider}: {str(exc)[:300]}")
            print(f"[AI] Provider {provider} failed; attempting next configured provider.")

    details = " | ".join(failures)
    if any(_is_retryable_error(Exception(item)) for item in failures):
        raise HTTPException(
            503,
            "All configured AI providers are temporarily unavailable or rate-limited. "
            "Please retry shortly. Provider details: " + details,
        )
    raise HTTPException(502, "All configured AI providers failed. " + details)


# New descriptive alias; existing routes continue using ask_gemini to avoid churn.
ask_ai = ask_gemini


def _transcribe_with_groq(contents: bytes, filename: str, mime_type: str) -> str:
    """Optional Groq Whisper fallback. Disabled by default to avoid unexpected usage charges."""
    if not GROQ_AUDIO_FALLBACK or not GROQ_API_KEY or not _groq_http:
        raise RuntimeError("Groq audio fallback is not enabled.")
    last_error = None
    for attempt in range(MAX_ATTEMPTS_PER_MODEL):
        try:
            response = _groq_http.post(
                "audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                data={"model": GROQ_AUDIO_MODEL, "response_format": "json", "temperature": "0"},
                files={"file": (filename or "interview.webm", contents, mime_type)},
            )
            response.raise_for_status()
            transcript = (response.json().get("text") or "").strip()
            if not transcript:
                raise RuntimeError("Groq returned an empty audio transcript.")
            return transcript
        except Exception as exc:
            last_error = exc
            if attempt + 1 < MAX_ATTEMPTS_PER_MODEL and _is_retryable_error(exc):
                _backoff(attempt)
                continue
            break
    raise RuntimeError(f"Groq audio transcription failed: {last_error}") from last_error


def _transcribe_with_gemini(contents: bytes, filename: str, mime_type: str) -> str:
    """Use Gemini first; optionally fall back to Groq Whisper when explicitly enabled."""
    failures = []
    if gemini_client:
        try:
            from google.genai import types
            audio_parts = [
                types.Part.from_bytes(data=contents, mime_type=mime_type),
                "Transcribe the spoken words verbatim. Return only the transcript; do not summarize or evaluate.",
            ]
            response = _gemini_generate(audio_parts)
            transcript = (response.text or "").strip()
            if transcript:
                return transcript
            raise RuntimeError("Gemini returned an empty audio transcript.")
        except Exception as exc:
            failures.append(f"Gemini: {str(exc)[:220]}")
            print("[AI] Gemini audio transcription failed; checking optional Groq audio fallback.")

    if GROQ_AUDIO_FALLBACK and GROQ_API_KEY:
        try:
            return _transcribe_with_groq(contents, filename, mime_type)
        except Exception as exc:
            failures.append(f"Groq audio: {str(exc)[:220]}")

    if not failures:
        raise HTTPException(
            503,
            "Audio transcription requires GEMINI_API_KEY. Groq audio fallback is disabled by default; "
            "you can provide a transcript or explicitly enable GROQ_AUDIO_FALLBACK after checking billing.",
        )
    raise HTTPException(
        503,
        "Audio transcription providers are unavailable. " + " | ".join(failures),
    )


@app.get("/")
def home():
    providers = _available_providers()
    return {
        "app": "CareerPilot",
        "status": "running",
        "model": MODEL,
        "providers": providers,
        "docs": "/docs",
    }

@app.get("/api/health")
def health():
    providers = _available_providers()
    return {
        "ok": True,
        "gemini_configured": bool(gemini_client),
        "groq_configured": bool(GROQ_API_KEY),
        "providers": providers,
        "primary_provider": providers[0] if providers else None,
        "model": MODEL,
        "groq_model": GROQ_MODEL,
        "groq_audio_fallback_enabled": GROQ_AUDIO_FALLBACK,
        "gemini_fallback_models": GEMINI_FALLBACK_MODELS,
    }

@app.post("/api/job/analyze")
async def analyze_job(resume: UploadFile = File(...), job_description: str = Form(...)):
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload your resume as a PDF.")
    if not job_description.strip():
        raise HTTPException(400, "Please provide a job description.")
    text = extract_pdf_text(await resume.read())
    prompt = f"""You are CareerPilot's job assistant. Analyze this resume against this job description.
Use ONLY facts supported by the resume and job description. Never invent candidate experience.
Classify skills as strong_matches, partial_matches, or missing_skills. Keep match_score realistic.
RESUME:\n{text[:25000]}\n\nJOB DESCRIPTION:\n{job_description[:20000]}"""
    return await asyncio.to_thread(ask_gemini, prompt, JobAnalysis)

@app.post("/api/job/application")
async def application(resume: UploadFile = File(...), job_description: str = Form(...)):
    text = extract_pdf_text(await resume.read())
    prompt = f"""You are CareerPilot's job application assistant.
Create truthful application materials using ONLY the resume and job description.
Never fabricate experience, employers, years, skills, achievements or certifications.
The application description must be 100-150 words and the cover letter 250-350 words.
RESUME:\n{text[:25000]}\n\nJOB DESCRIPTION:\n{job_description[:20000]}"""
    return await asyncio.to_thread(ask_gemini, prompt, ApplicationMaterials)

@app.post("/api/job/roles")
async def roles(resume: UploadFile = File(...)):
    text = extract_pdf_text(await resume.read())
    prompt = f"""You are a job recommendation engine. Based ONLY on this resume, recommend exactly 5 realistic
entry-level or internship roles. Do not invent experience. Fit scores must reflect actual evidence in the resume.
For skills_to_improve, list skills that would genuinely help the candidate qualify for that role.
RESUME:\n{text[:25000]}"""
    return await asyncio.to_thread(ask_gemini, prompt, RoleRecommendations)

@app.post("/api/interview/questions")
async def questions(role: str = Form(...), job_description: str = Form(""),
                    interview_type: str = Form("Technical + HR"), difficulty: str = Form("Medium")):
    prompt = f"""Create a realistic interview plan for a student/fresher.
Generate exactly 10 questions. Mix categories appropriately. Follow-up questions later will depend on answers.
ROLE: {role}\nTYPE: {interview_type}\nDIFFICULTY: {difficulty}\nJOB DESCRIPTION: {job_description[:15000]}"""
    return await asyncio.to_thread(ask_gemini, prompt, InterviewPlan)

@app.post("/api/interview/evaluate")
async def evaluate(role: str = Form(...), question: str = Form(...), answer: str = Form(...)):
    prompt = f"""Evaluate this interview answer fairly. Judge only the content of the answer and its fit to the question.
Do not infer personality, mental state, honesty or hidden emotions.
Category scores are 0-10; overall_score is 0-100.
ROLE: {role}\nQUESTION: {question}\nANSWER: {answer}"""
    return await asyncio.to_thread(ask_gemini, prompt, InterviewEvaluation)

@app.post("/api/interview/followup")
async def followup(role: str = Form(...), question: str = Form(...), answer: str = Form(...)):
    prompt = f"""Act as a realistic interviewer. Ask exactly ONE natural follow-up based on something actually said by the candidate.
Do not invent a claim that the candidate did not make. If the answer is vague, ask a clarification/probing question.
ROLE: {role}\nQUESTION: {question}\nANSWER: {answer}"""
    return await asyncio.to_thread(ask_gemini, prompt, Followup)
async def _transcribe_audio(contents: bytes, filename: str) -> str:
    """Transcribe interview audio with Gemini and retry temporary failures."""
    suffix = os.path.splitext(filename or "")[1].lower()
    mime_types = {
        ".webm": "audio/webm", ".wav": "audio/wav", ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
        ".flac": "audio/flac",
    }
    mime_type = mime_types.get(suffix, "audio/webm")
    return await asyncio.to_thread(_transcribe_with_gemini, contents, filename, mime_type)


async def _save_and_analyze_audio(audio: UploadFile, transcript: str = "") -> dict:
    contents = await audio.read()
    if not contents:
        raise HTTPException(400, "The uploaded audio is empty.")
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(413, "Recording is too large. Keep interview answers under 25 MB.")
    suffix = os.path.splitext(audio.filename or "")[1].lower() or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name
    try:
        if not transcript.strip():
            transcript = await _transcribe_audio(contents, audio.filename or suffix)
        # Librosa feature extraction is CPU-bound; keep it off the event loop.
        metrics = await asyncio.to_thread(analyze_audio, temp_path, transcript)
        return {"success": True, "transcript": transcript, "speech_metrics": metrics}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/interview/audio")
@app.post("/api/interview/audio/analyze")
async def analyze_interview_audio(audio: UploadFile = File(...), transcript: str = Form("")):
    return await _save_and_analyze_audio(audio, transcript)


@app.post("/api/interview/audio/evaluate")
async def evaluate_interview_audio(
    audio: UploadFile = File(...),
    question: str = Form(...),
    role: str = Form("General interview"),
    transcript: str = Form(""),
):
    analyzed = await _save_and_analyze_audio(audio, transcript)
    evaluation_prompt = (
        f"Evaluate this mock-interview answer fairly using only the transcript. Do not infer personality, mental state, honesty, or emotion. Score categories 0-10 and overall_score 0-100.\n"
        f"ROLE: {role}\nQUESTION: {question}\nTRANSCRIPT: {analyzed['transcript']}"
    )
    evaluation = await asyncio.to_thread(ask_gemini, evaluation_prompt, InterviewEvaluation)
    return {**analyzed, "evaluation": evaluation}

# Phase 4: optional camera frame analysis. Frames are processed in memory, not saved.
@app.post("/api/interview/vision/frame")
async def vision_frame(frame: UploadFile = File(...)):
    from .vision_analysis import analyze_frame
    try:
        frame_bytes = await frame.read()
        vision = await asyncio.to_thread(analyze_frame, frame_bytes)
        return {"success": True, "vision": vision}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)[:300]) from e

# Phase 5: session-level summary. Frames and audio metrics should be aggregated by the frontend.
@app.post("/api/interview/final-report")
async def final_report(payload: dict):
    prompt = f'''Return a balanced final mock-interview report as JSON matching the requested schema.
Role: {payload.get("role", "unspecified")}
Answer evaluations and transcripts: {str(payload.get("answers", []))[:16000]}
Speech metrics: {str(payload.get("speech_summary", []))[:5000]}
Observable camera summary: {str(payload.get("vision_summary", {}))[:3000]}
Base content scores on answer evaluations and delivery notes on measurable speech metrics. Camera data may describe only face visibility and approximate head orientation. Never infer confidence, emotion, personality, honesty, or competence from video. If data is missing, say so. Give 3-5 practical personalized practice steps.'''
    result = await asyncio.to_thread(ask_gemini, prompt, FinalInterviewReport)
    return {"success": True, "report": result}
