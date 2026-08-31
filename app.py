"""
app.py
------
Streamlit user interface for the AI ATS Resume Analyzer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ats_analyzer import analyze_resume_against_jd, llm_detailed_analysis
from resume_parser import ResumeParseError, extract_text_from_pdf


st.set_page_config(
    page_title="AI ATS Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef2f7 100%);
    }
    h1, h2, h3 {
        color: #0f3d68;
        letter-spacing: -0.02em;
    }
    .hero {
        background: linear-gradient(135deg, #0f3d68 0%, #1a6aa5 60%, #1d9a8a 100%);
        color: #ffffff;
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 61, 104, 0.18);
    }
    .hero h1 {
        color: #ffffff;
        margin: 0 0 0.4rem 0;
        font-size: 1.85rem;
    }
    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }
    .score-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        border: 1px solid #dbe4ef;
        box-shadow: 0 8px 24px rgba(15, 61, 104, 0.06);
    }
    .score-number {
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1;
        color: #0f3d68;
    }
    .score-label {
        color: #5b6b7c;
        margin-top: 0.35rem;
        font-size: 0.95rem;
    }
    .pill {
        display: inline-block;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        margin: 0.18rem 0.28rem 0.18rem 0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pill-ok {
        background: #e5f7f2;
        color: #0f6b57;
        border: 1px solid #b7e6d8;
    }
    .pill-miss {
        background: #fdecea;
        color: #9b2c2c;
        border: 1px solid #f5c6c2;
    }
    .pill-skill {
        background: #e8f1fb;
        color: #1a4d80;
        border: 1px solid #c5daf3;
    }
    .muted {
        color: #5b6b7c;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dbe4ef;
        padding: 0.8rem 0.9rem;
        border-radius: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def score_band(score: int) -> tuple[str, str]:
    if score >= 75:
        return "Strong match", "#1d9a8a"
    if score >= 55:
        return "Moderate match", "#d97706"
    return "Needs work", "#c2410c"


def render_pills(items: list[str], kind: str) -> None:
    if not items:
        st.caption("None detected from the provided text.")
        return
    css = {"ok": "pill-ok", "miss": "pill-miss", "skill": "pill-skill"}[kind]
    html = "".join(f'<span class="pill {css}">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>AI ATS Resume Analyzer</h1>
            <p>
                Upload a text-based PDF resume and paste a job description.
                Get an ATS-style score, keyword gaps, formatting checks, and optional AI feedback.
                This tool never invents experience — it only uses text from your files.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("How scoring works")
        st.markdown(
            """
            **ATS score / 100**
            - **40 pts** — keyword match vs the job post
            - **35 pts** — TF-IDF cosine similarity
            - **25 pts** — ATS formatting checks

            **Tips**
            - Use a simple, single-column PDF
            - Do not add skills you do not have
            - AI analysis needs `OPENAI_API_KEY` in `.env`
            """
        )
        st.divider()
        st.caption("Built with Python, Streamlit, scikit-learn, pypdf, and an optional LLM API.")

    col_left, col_right = st.columns((1, 1), gap="large")

    with col_left:
        st.subheader("1. Resume PDF")
        uploaded = st.file_uploader(
            "Upload your resume",
            type=["pdf"],
            help="Text-based PDFs work best. Scanned photos usually cannot be parsed.",
        )

    with col_right:
        st.subheader("2. Job description")
        job_description = st.text_area(
            "Paste the full job description",
            height=220,
            placeholder="Paste the job posting here…",
        )

    run_llm = st.checkbox(
        "Include detailed AI analysis (uses your API key from .env)",
        value=True,
    )
    analyze_clicked = st.button("Analyze resume", type="primary", use_container_width=True)

    if not analyze_clicked:
        st.info("Upload a PDF and paste a job description, then click **Analyze resume**.")
        return

    if uploaded is None:
        st.error("Please upload a resume PDF.")
        return
    if not (job_description or "").strip():
        st.error("Please paste a job description.")
        return
    if len(job_description.strip()) < 40:
        st.error("The job description is too short to analyze. Paste the full posting.")
        return

    try:
        with st.spinner("Extracting text from the PDF…"):
            resume_text = extract_text_from_pdf(uploaded)
    except ResumeParseError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Could not read the PDF: {exc}")
        return

    with st.spinner("Comparing resume with the job description…"):
        try:
            analysis = analyze_resume_against_jd(resume_text, job_description)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            return

    score = analysis["ats_score"]
    band, color = score_band(score)
    parts = analysis["score_parts"]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-number" style="color:{color}">{score}</div>
                <div class="score-label">ATS score / 100<br><strong>{band}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    m2.metric("Keyword match", f"{analysis['keyword_match_percent']}%")
    m3.metric("TF-IDF similarity", f"{analysis['cosine_similarity']}%")
    m4.metric("Formatting score", f"{analysis['format_score']}/100")

    st.caption(
        f"Score mix: keywords {parts['keyword_points']}/{parts['keyword_max']} · "
        f"similarity {parts['similarity_points']}/{parts['similarity_max']} · "
        f"formatting {parts['format_points']}/{parts['format_max']}"
    )

    st.divider()
    k1, k2 = st.columns(2)
    with k1:
        st.subheader("Matched keywords")
        st.markdown(
            f'<p class="muted">{len(analysis["matched_keywords"])} important job terms found in the resume.</p>',
            unsafe_allow_html=True,
        )
        render_pills(analysis["matched_keywords"], "ok")
    with k2:
        st.subheader("Missing keywords")
        st.markdown(
            f'<p class="muted">{len(analysis["missing_keywords"])} job terms not found as written in the resume.</p>',
            unsafe_allow_html=True,
        )
        render_pills(analysis["missing_keywords"], "miss")

    st.subheader("Technical skills found in the resume")
    st.markdown(
        '<p class="muted">Only skills that already appear in your PDF are listed.</p>',
        unsafe_allow_html=True,
    )
    render_pills(analysis["technical_skills"], "skill")

    s1, s2 = st.columns(2)
    with s1:
        st.subheader("Strengths")
        for item in analysis["strengths"]:
            st.success(item)
    with s2:
        st.subheader("Weaknesses")
        for item in analysis["weaknesses"]:
            st.warning(item)

    st.subheader("ATS formatting")
    if analysis["formatting_ok"]:
        for note in analysis["formatting_ok"]:
            st.write(f"- {note}")
    if analysis["formatting_issues"]:
        for issue in analysis["formatting_issues"]:
            st.error(issue)
    else:
        st.success("No major formatting issues were detected in the extracted text.")

    st.subheader("Improvement suggestions")
    for i, tip in enumerate(analysis["suggestions"], start=1):
        st.write(f"{i}. {tip}")

    with st.expander("View extracted resume text"):
        st.text(resume_text)

    if run_llm:
        st.divider()
        st.subheader("Detailed AI analysis")
        with st.spinner("Asking the language model for feedback…"):
            llm = llm_detailed_analysis(resume_text, job_description, analysis)
        if not llm["ok"]:
            st.warning(llm["error"])
        else:
            data = llm["data"]
            if data.get("overview"):
                st.write(data["overview"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**AI — strengths**")
                for item in data.get("strengths") or []:
                    st.write(f"- {item}")
            with c2:
                st.markdown("**AI — weaknesses**")
                for item in data.get("weaknesses") or []:
                    st.write(f"- {item}")
            if data.get("keyword_advice"):
                st.markdown("**Keyword advice**")
                st.write(data["keyword_advice"])
            if data.get("formatting_advice"):
                st.markdown("**Formatting advice**")
                st.write(data["formatting_advice"])
            if data.get("suggestions"):
                st.markdown("**AI — extra suggestions**")
                for item in data["suggestions"]:
                    st.write(f"- {item}")

    st.caption(
        "Disclaimer: This is an educational ATS-style checker. Real applicant tracking "
        "systems differ. Never add false experience to raise a score."
    )


if __name__ == "__main__":
    main()
