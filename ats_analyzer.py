"""
ats_analyzer.py
---------------
Builds an understandable ATS score and optional LLM analysis.

Score breakdown (100 points total):
  40  Keyword match          – how many important JD terms appear in the resume
  35  TF-IDF similarity      – overall wording overlap (cosine similarity)
  25  ATS formatting         – whether the PDF is likely easy for an ATS to parse

The LLM is used only to explain the extracted text. Prompts tell the model
not to invent jobs, degrees, dates, or skills.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

from keyword_matcher import (
    cosine_similarity_score,
    detect_technical_skills,
    extract_jd_keywords,
    keyword_match,
)
from resume_parser import detect_section_headings, looks_like_email, looks_like_phone

load_dotenv()


def analyze_resume_against_jd(resume_text: str, job_description: str) -> dict[str, Any]:
    """
    Run the full rule-based analysis (no API required).

    Returns a dictionary the Streamlit app can display directly.
    """
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()

    similarity = cosine_similarity_score(resume_text, job_description)
    jd_keywords = extract_jd_keywords(job_description)
    keyword_result = keyword_match(resume_text, jd_keywords)
    skills = detect_technical_skills(resume_text)
    formatting = detect_formatting_issues(resume_text)
    strengths, weaknesses = rule_based_strengths_weaknesses(
        resume_text=resume_text,
        keyword_result=keyword_result,
        similarity=similarity,
        skills=skills,
        formatting=formatting,
    )
    suggestions = rule_based_suggestions(
        keyword_result=keyword_result,
        formatting=formatting,
        skills=skills,
        resume_text=resume_text,
    )

    keyword_points = (keyword_result["match_percent"] / 100.0) * 40.0
    similarity_points = similarity * 35.0
    format_points = (formatting["format_score"] / 100.0) * 25.0
    ats_score = round(keyword_points + similarity_points + format_points)

    return {
        "ats_score": int(max(0, min(100, ats_score))),
        "score_parts": {
            "keyword_points": round(keyword_points, 1),
            "similarity_points": round(similarity_points, 1),
            "format_points": round(format_points, 1),
            "keyword_max": 40,
            "similarity_max": 35,
            "format_max": 25,
        },
        "keyword_match_percent": keyword_result["match_percent"],
        "matched_keywords": keyword_result["matched"],
        "missing_keywords": keyword_result["missing"],
        "jd_keywords": jd_keywords,
        "cosine_similarity": round(similarity * 100, 1),
        "technical_skills": skills,
        "formatting_issues": formatting["issues"],
        "formatting_ok": formatting["ok_notes"],
        "format_score": formatting["format_score"],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "sections_found": detect_section_headings(resume_text),
    }


def detect_formatting_issues(resume_text: str) -> dict[str, Any]:
    """
    Heuristic ATS-friendliness checks based on extracted text.

    These are common ATS problems, not a guarantee of how every ATS behaves.
    """
    issues: list[str] = []
    ok_notes: list[str] = []
    deductions = 0

    length = len(resume_text)

    if length < 400:
        issues.append(
            "The extracted text is very short. ATS tools often fail on image-based "
            "or heavily designed PDFs. Prefer a simple, text-based layout."
        )
        deductions += 30
    else:
        ok_notes.append("Enough text was extracted for an ATS to read the resume.")

    if not looks_like_email(resume_text):
        issues.append(
            "No email address was found in the extracted text. Put your email in "
            "plain text in the header (not inside an image or icon)."
        )
        deductions += 15
    else:
        ok_notes.append("An email address appears in the text.")

    if not looks_like_phone(resume_text):
        issues.append(
            "No phone number was found. Add a phone number in plain text so ATS "
            "and recruiters can contact you."
        )
        deductions += 10
    else:
        ok_notes.append("A phone number appears in the text.")

    sections = detect_section_headings(resume_text)
    if len(sections) < 2:
        issues.append(
            "Standard section headings (Experience, Education, Skills, etc.) were "
            "hard to detect. Use clear headings on their own lines."
        )
        deductions += 15
    else:
        ok_notes.append(f"Detected section headings: {', '.join(sections)}.")

    special_ratio = _unusual_character_ratio(resume_text)
    if special_ratio > 0.08:
        issues.append(
            "The text has many unusual symbols. Fancy fonts, icons, and text boxes "
            "often confuse ATS parsers. Stick to common fonts and simple bullets."
        )
        deductions += 10

    if re.search(r"\|.+\|", resume_text):
        issues.append(
            "Pipe characters (|) were found, which often come from tables or "
            "multi-column layouts. Many ATS systems read tables poorly. Use a "
            "single-column layout when possible."
        )
        deductions += 10

    bullet_count = len(re.findall(r"(?m)^\s*[-•●▪◦*]\s+", resume_text))
    if bullet_count < 3:
        issues.append(
            "Few bullet points were detected. Simple '-' or '•' bullets are easier "
            "for ATS tools than nested graphics or columns."
        )
        deductions += 5
    else:
        ok_notes.append("Bullet points were detected in the text.")

    format_score = max(0, 100 - deductions)
    return {
        "issues": issues,
        "ok_notes": ok_notes,
        "format_score": format_score,
    }


def rule_based_strengths_weaknesses(
    resume_text: str,
    keyword_result: dict,
    similarity: float,
    skills: list[str],
    formatting: dict,
) -> tuple[list[str], list[str]]:
    """Strengths and weaknesses based only on the uploaded text and JD match."""
    strengths: list[str] = []
    weaknesses: list[str] = []

    if keyword_result["match_percent"] >= 60:
        strengths.append(
            f"Keyword coverage is solid ({keyword_result['match_percent']}% of "
            "important job-description terms appear in the resume)."
        )
    elif keyword_result["match_percent"] < 35:
        weaknesses.append(
            f"Keyword coverage is low ({keyword_result['match_percent']}%). "
            "Many job-description terms do not appear in the resume."
        )

    if similarity >= 0.25:
        strengths.append(
            "The resume and job description share a noticeable amount of wording "
            f"(TF-IDF similarity {round(similarity * 100, 1)}%)."
        )
    elif similarity < 0.12:
        weaknesses.append(
            "Overall wording overlap with the job description is low. The resume "
            "may be written for a different role or industry."
        )

    if len(skills) >= 8:
        strengths.append(
            f"Several technical skills are listed in the resume ({len(skills)} detected)."
        )
    elif len(skills) <= 2:
        weaknesses.append(
            "Few well-known technical skills were detected as plain text. If you "
            "have relevant skills, write them out (for example in a Skills section)."
        )

    if re.search(r"\b\d+%|\b\d+\+|increased|reduced|improved|saved\b", resume_text, re.I):
        strengths.append(
            "The resume includes numbers or result language, which helps describe impact."
        )
    else:
        weaknesses.append(
            "Little quantified impact was found (percentages, counts, or result verbs). "
            "Where true, add results you actually achieved — do not invent metrics."
        )

    if formatting["issues"]:
        weaknesses.append(
            f"{len(formatting['issues'])} ATS formatting concern(s) were flagged. "
            "See the formatting section for details."
        )
    else:
        strengths.append("No major ATS formatting red flags were found in the extracted text.")

    if not strengths:
        strengths.append(
            "The resume could be parsed as text. Use the missing-keyword list to "
            "align it more closely with this job description."
        )
    return strengths, weaknesses


def rule_based_suggestions(
    keyword_result: dict,
    formatting: dict,
    skills: list[str],
    resume_text: str,
) -> list[str]:
    """Practical, honest suggestions. Never tells the user to claim false skills."""
    suggestions: list[str] = []

    missing = keyword_result.get("missing") or []
    if missing:
        preview = ", ".join(missing[:8])
        extra = f" (and {len(missing) - 8} more)" if len(missing) > 8 else ""
        suggestions.append(
            "If you truly have experience with these job terms, add them in natural "
            f"sentences or a Skills section: {preview}{extra}. "
            "Do not copy keywords you cannot support in an interview."
        )

    if formatting.get("issues"):
        suggestions.append(
            "Export a simple, single-column PDF (no tables, text boxes, or icons "
            "for contact details) so ATS software can read every line."
        )

    if "skills" not in resume_text.lower():
        suggestions.append(
            "Add a clearly titled Skills section so both humans and ATS tools can "
            "find your tools and technologies quickly."
        )

    if len(skills) > 0:
        suggestions.append(
            "Mirror the job description's wording for skills you already have "
            "(for example 'REST API' vs 'RESTful APIs') — only when it is accurate."
        )

    suggestions.append(
        "Tailor the top third of the resume (summary and recent role) to this job. "
        "Keep facts truthful: titles, dates, and employers must stay accurate."
    )
    return suggestions


def llm_detailed_analysis(
    resume_text: str,
    job_description: str,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Call an OpenAI-compatible chat API for a deeper write-up.

    Returns:
        {"ok": True, "data": {...}} or {"ok": False, "error": "..."}.

    The model is instructed not to invent candidate information.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False,
            "error": (
                "No OPENAI_API_KEY found. Copy .env.example to .env and add your key "
                "to enable AI analysis. Keyword matching and ATS scoring still work."
            ),
        }

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    payload = {
        "ats_score": analysis.get("ats_score"),
        "keyword_match_percent": analysis.get("keyword_match_percent"),
        "cosine_similarity_percent": analysis.get("cosine_similarity"),
        "matched_keywords": analysis.get("matched_keywords"),
        "missing_keywords": analysis.get("missing_keywords"),
        "technical_skills_found_in_resume": analysis.get("technical_skills"),
        "formatting_issues": analysis.get("formatting_issues"),
        "sections_found": analysis.get("sections_found"),
    }

    system = (
        "You are an ATS resume coach. You only discuss text the user provided. "
        "Never invent employers, job titles, dates, degrees, certifications, "
        "metrics, or skills. If something is not in the resume, say it is missing. "
        "Do not tell the candidate to claim experience they did not write. "
        "Be specific, kind, and practical. Return valid JSON only."
    )
    user = (
        "Compare this resume to this job description.\n\n"
        "=== RESUME TEXT (extracted from PDF) ===\n"
        f"{resume_text[:12000]}\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{job_description[:8000]}\n\n"
        "=== RULE-BASED STATS (for context; you may reference them) ===\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "overview": "3-6 sentence honest overview",\n'
        '  "strengths": ["...", "..."],\n'
        '  "weaknesses": ["...", "..."],\n'
        '  "suggestions": ["...", "..."],\n'
        '  "keyword_advice": "how to add missing keywords truthfully",\n'
        '  "formatting_advice": "ATS formatting advice based on the extracted text"\n'
        "}\n"
        "Use 4 to 7 items in each of strengths, weaknesses, and suggestions."
    )

    try:
        from openai import OpenAI
    except ImportError:
        return {
            "ok": False,
            "error": "The openai package is not installed. Run: pip install -r requirements.txt",
        }

    try:
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        data = _parse_json_object(content)
        if data is None:
            return {
                "ok": False,
                "error": "The AI returned a response that was not valid JSON. Try again.",
            }
        return {"ok": True, "data": data}
    except Exception as exc:
        return {"ok": False, "error": f"AI analysis failed: {exc}"}


def _parse_json_object(text: str) -> Optional[dict]:
    """Extract a JSON object from a model reply that may include markdown fences."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def _unusual_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    unusual = sum(1 for ch in text if ord(ch) > 127 and ch not in "•–—’“”éèáüö")
    return unusual / max(len(text), 1)
