"""
keyword_matcher.py
------------------
Compares resume text with a job description using:

1. TF-IDF + cosine similarity (overall document match)
2. Keyword overlap (which JD terms appear in the resume)
3. A technical-skills dictionary (skills found in the resume)

Nothing here invents skills or experience. A keyword is "matched"
only if it literally appears in the resume text.
"""

from __future__ import annotations

import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Common technical skills. Used only to *detect* skills already written
# in the resume. The app never adds skills that are not in the document.
TECHNICAL_SKILLS: tuple[str, ...] = (
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "ruby", "php", "swift", "kotlin", "r", "scala", "rust", "matlab", "sql",
    "html", "css", "bash", "shell",
    # Data / ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data science", "data engineering",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "keras", "huggingface", "opencv", "spark", "pyspark", "hadoop", "airflow",
    "tableau", "power bi", "excel", "statistics", "a/b testing",
    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "jenkins",
    "ci/cd", "terraform", "ansible", "linux", "git", "github", "gitlab",
    # Web / backend
    "react", "angular", "vue", "node.js", "nodejs", "express", "django",
    "flask", "fastapi", "streamlit", "spring", "rest api", "graphql",
    "microservices",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "oracle",
    "sqlite", "snowflake", "redshift", "elasticsearch",
    # Other
    "jira", "agile", "scrum", "kanban", "unit testing", "pytest",
    "selenium", "api", "json", "xml", "oauth", "jwt",
)

# Extra stop words that are common in job posts but not useful as ATS keywords.
_EXTRA_STOP = {
    "experience", "experiences", "year", "years", "role", "roles", "job",
    "position", "candidate", "candidates", "team", "teams", "work", "working",
    "ability", "able", "including", "required", "requirements", "preferred",
    "plus", "must", "using", "used", "use", "strong", "good", "great",
    "excellent", "knowledge", "understanding", "skills", "skill", "etc",
    "company", "opportunity", "responsibilities", "responsibility",
}


def cosine_similarity_score(resume_text: str, job_description: str) -> float:
    """
    Return a 0–1 cosine similarity between resume and job description
    using TF-IDF word and phrase weights.

    0.0 = almost no overlapping important terms
    1.0 = the two texts use very similar important terms
    """
    resume_text = (resume_text or "").strip()
    job_description = (job_description or "").strip()
    if not resume_text or not job_description:
        return 0.0

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
    )
    try:
        matrix = vectorizer.fit_transform([resume_text, job_description])
    except ValueError:
        return 0.0

    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    # Guard against tiny floating-point noise.
    return float(max(0.0, min(1.0, similarity)))


def extract_jd_keywords(job_description: str, top_n: int = 35) -> list[str]:
    """
    Pick the most important terms from the job description using TF-IDF.

    Only the job description is used (a single document). TF-IDF still
    down-weights very common English words via stop-word removal, and
    n-grams help capture phrases like "machine learning".
    """
    job_description = (job_description or "").strip()
    if not job_description:
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b",
        max_features=800,
    )
    try:
        matrix = vectorizer.fit_transform([job_description])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    weights = matrix.toarray()[0]
    ranked = sorted(zip(feature_names, weights), key=lambda item: item[1], reverse=True)

    stop = set(ENGLISH_STOP_WORDS) | _EXTRA_STOP
    keywords: list[str] = []
    for term, weight in ranked:
        if weight <= 0:
            continue
        if term in stop:
            continue
        if len(term) < 2:
            continue
        # Skip terms that are only digits/punctuation after cleanup.
        if not re.search(r"[a-zA-Z]", term):
            continue
        keywords.append(term)
        if len(keywords) >= top_n:
            break
    return keywords


def keyword_match(resume_text: str, keywords: Iterable[str]) -> dict:
    """
    Check which keywords appear in the resume as whole phrases.

    Returns
    -------
    dict with keys:
        matched, missing, match_percent, total
    """
    resume_lower = (resume_text or "").lower()
    matched: list[str] = []
    missing: list[str] = []

    seen = set()
    for raw in keywords:
        keyword = (raw or "").strip()
        if not keyword:
            continue
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        if _phrase_in_text(key, resume_lower):
            matched.append(keyword)
        else:
            missing.append(keyword)

    total = len(matched) + len(missing)
    percent = round((len(matched) / total) * 100, 1) if total else 0.0
    return {
        "matched": matched,
        "missing": missing,
        "match_percent": percent,
        "total": total,
    }


def detect_technical_skills(resume_text: str) -> list[str]:
    """
    Return technical skills from TECHNICAL_SKILLS that appear in the resume.
    Skills not written in the resume are not listed.
    """
    resume_lower = (resume_text or "").lower()
    found: list[str] = []
    for skill in TECHNICAL_SKILLS:
        if _phrase_in_text(skill, resume_lower):
            found.append(skill)
    # Keep longer / more specific names first, then alphabetical.
    found.sort(key=lambda s: (-len(s), s))
    return found


def _phrase_in_text(phrase: str, text_lower: str) -> bool:
    """
    True if the phrase appears in text. Short tokens (sql, r, go, aws)
    are matched as whole words so they are not found inside other words.
    """
    phrase = phrase.lower().strip()
    if not phrase:
        return False
    if len(phrase) <= 3 or " " not in phrase:
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
        return re.search(pattern, text_lower) is not None
    return phrase in text_lower
