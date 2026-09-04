# AI ATS Resume Analyzer

A beginner-friendly Streamlit app that compares a **resume PDF** with a **job description**, estimates an **ATS-style score**, and optionally asks an LLM for detailed feedback.

The app **does not invent** jobs, degrees, skills, or metrics. It only uses text extracted from your PDF and the job description you paste.

## Features

- Upload a resume PDF
- Extract text with **pypdf**
- Paste a job description
- **TF-IDF** + **cosine similarity** for overall match
- Keyword match percentage
- Matched and missing keywords
- Technical skills detected in the resume text
- Strengths, weaknesses, and improvement suggestions
- ATS formatting checks (email, phone, headings, tables, scanned PDFs)
- Optional **LLM** analysis via an API key in `.env`

## Project files

| File | Role |
|------|------|
| `app.py` | Streamlit UI. Upload PDF, paste JD, show results. |
| `resume_parser.py` | Reads the PDF and extracts plain text. |
| `keyword_matcher.py` | TF-IDF similarity, keyword overlap, skill detection. |
| `ats_analyzer.py` | ATS score (out of 100), formatting checks, LLM call. |
| `requirements.txt` | Python packages to install. |
| `.env.example` | Template for your API key (copy to `.env`). |
| `.gitignore` | Ignores `.env`, virtualenvs, and cache files. |

## How the ATS score is calculated

The score is out of **100** and is easy to explain:

| Part | Points | Meaning |
|------|--------|---------|
| Keyword match | **40** | Share of important job-description terms that appear in the resume |
| TF-IDF cosine similarity | **35** | How similar the two documents are as a whole |
| ATS formatting | **25** | Whether the extracted text looks ATS-friendly |

Example: 70% keyword match → 28 points, 40% similarity → 14 points, formatting 80/100 → 20 points. **Total = 62**.

A higher score means closer wording and cleaner parsing. It is **not** a hiring decision and is **not** identical to any one company's ATS.

## Setup

You need **Python 3.10+**.

```bash
cd AI_resume
python -m venv .venv
```

**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## API key (optional but needed for AI analysis)

1. Copy the example env file:

   ```bash
   copy .env.example .env
   ```

   On macOS/Linux: `cp .env.example .env`

2. Open `.env` and set `OPENAI_API_KEY`. Do not put the key in any Python file.

3. You can change `OPENAI_MODEL` or set `OPENAI_BASE_URL` for an OpenAI-compatible API.

Keyword matching and scoring **work without** an API key. Uncheck “Include detailed AI analysis” or leave the key empty if you only want the rule-based report.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How to use it

1. Upload a **text-based** PDF resume (not a photo scan).
2. Paste the **full** job description.
3. Click **Analyze resume**.
4. Review score, keywords, skills, formatting, and suggestions.
5. Only add missing keywords that reflect **real** experience.

## Notes for beginners

- If extraction fails, the PDF is likely scanned or image-heavy. Export again from Word/Google Docs as a simple PDF.
- Missing keywords are terms from the job post that were **not found as written**. Synonyms may still exist in the resume.
- The technical-skills list is a helper dictionary. Skills not on that list can still appear under matched keywords if they are in the job description.

## L
