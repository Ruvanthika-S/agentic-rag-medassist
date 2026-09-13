"""MedAssist Streamlit interface for the real agentic RAG orchestrator."""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.orchestrator import Orchestrator


st.set_page_config(
    page_title="MedAssist | Grounded Biomedical QA",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --ink: #e8f1f3;
    --muted: #91a8ad;
    --panel: #101d21;
    --panel-soft: #14252a;
    --line: rgba(154, 203, 204, .15);
    --teal: #61d4c8;
    --cyan: #8bd8f4;
    --amber: #f1be63;
    --red: #ff8c7d;
    --green: #83e1ad;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 85% 0%, #173c43 0, #0a1417 38%, #071012 100%); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0b171a; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem; }
.block-container { max-width: 1220px; padding: 3.5rem 3rem 5rem; }

.hero { padding: 1rem 0 2.25rem; border-bottom: 1px solid var(--line); margin-bottom: 2rem; }
.eyebrow { color: var(--teal); font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.8rem, 6vw, 5.4rem); line-height: .95; letter-spacing: -.04em; margin: .45rem 0 .8rem; background: linear-gradient(110deg, #f4ffff 15%, var(--teal) 58%, var(--cyan)); -webkit-background-clip: text; color: transparent; }
.hero p { color: var(--muted); font-size: 1.12rem; margin: 0; }
.architecture { display: inline-block; margin-top: 1.15rem; padding: .42rem .72rem; border: 1px solid rgba(97, 212, 200, .3); border-radius: 999px; color: #b9eeea; background: rgba(97, 212, 200, .07); font-size: .78rem; }

.section-label { color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin: 1rem 0 .55rem; }
.answer-card { padding: 1.35rem 1.5rem; border: 1px solid var(--line); border-left: 4px solid var(--teal); border-radius: 14px; background: linear-gradient(135deg, rgba(22, 48, 52, .92), rgba(13, 26, 29, .96)); box-shadow: 0 18px 50px rgba(0,0,0,.18); }
.answer-card.no { border-left-color: var(--red); }
.answer-card.maybe { border-left-color: var(--amber); }
.answer-card.insufficient_evidence { border-left-color: #ffcf70; background: linear-gradient(135deg, rgba(67, 49, 25, .58), rgba(24, 25, 20, .96)); }
.answer-kicker { color: var(--muted); font-size: .74rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
.answer-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.7rem; font-weight: 700; text-transform: capitalize; margin-top: .25rem; }
.rationale { color: #c7d7da; font-size: 1.03rem; line-height: 1.7; padding: 1.1rem 0 .2rem; }
.metric { min-height: 92px; padding: 1rem 1.1rem; border: 1px solid var(--line); border-radius: 12px; background: rgba(16, 29, 33, .74); }
.metric-label { color: var(--muted); font-size: .73rem; text-transform: uppercase; letter-spacing: .1em; }
.metric-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 600; margin-top: .45rem; }
.evidence-card { padding: 1rem 1.15rem; margin: .65rem 0; border: 1px solid var(--line); border-radius: 10px; background: rgba(14, 27, 30, .7); }
.evidence-meta { color: var(--muted); font-size: .78rem; }
.evidence-meta a { color: var(--cyan); text-decoration: none; }
.evidence-text { color: #d2e0e2; line-height: 1.55; margin-top: .55rem; }
.score { color: #809b9f; font-size: .72rem; margin-top: .6rem; }
.flow { color: #c6d7da; line-height: 2; font-size: .88rem; }
.flow strong { color: var(--teal); font-weight: 600; }
.side-note { color: var(--muted); font-size: .84rem; line-height: 1.6; }
div[data-testid="stTextArea"] textarea { background: #102125; color: var(--ink); border: 1px solid rgba(139, 216, 244, .24); border-radius: 12px; font-size: 1rem; }
div.stButton > button { border-radius: 9px; border: 1px solid rgba(97, 212, 200, .4); background: linear-gradient(135deg, #1b827d, #276d82); color: white; font-weight: 700; }
div.stButton > button:hover { border-color: var(--teal); color: white; }
</style>
""",
    unsafe_allow_html=True,
)


EXAMPLES = [
    "Can losartan reduce brain atrophy in Alzheimer's disease?",
    "Is PRP-40 regulation of microexons a conserved phenomenon?",
    "What are the side effects of aspirin?",
    "Treatment for hypertension in adults",
]


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


def answer_class(answer: str) -> str:
    return answer if answer in {"yes", "no", "maybe", "insufficient_evidence"} else "maybe"


def render_evidence(evidence: list[dict]) -> None:
    if not evidence:
        st.info("No evidence sources were returned.")
        return
    for item in evidence:
        pmid = str(item.get("pmid") or "")
        source = html.escape(str(item.get("source_dataset") or "unknown"))
        text = html.escape(str(item.get("chunk_text") or ""))
        pmid_html = (
            f'<a href="https://pubmed.ncbi.nlm.nih.gov/{html.escape(pmid)}/" target="_blank">PMID {html.escape(pmid)}</a>'
            if pmid and pmid.isdigit()
            else "Web source"
        )
        st.markdown(
            f"""<div class="evidence-card">
<div class="evidence-meta">{pmid_html} &nbsp; · &nbsp; <strong>{source}</strong></div>
<div class="evidence-text">{text}</div>
<div class="score">BM25 {float(item.get('bm25_score', 0)):.3f} &nbsp; · &nbsp; Dense {float(item.get('faiss_score', 0)):.4f} &nbsp; · &nbsp; Fusion {float(item.get('combined_score', 0)):.5f}</div>
</div>""",
            unsafe_allow_html=True,
        )


with st.sidebar:
    st.markdown('<div class="eyebrow">System map</div><h2>How it works</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="flow"><strong>Retriever</strong><br>↓<br><strong>Confidence check</strong><br>↓<br><strong>Web fallback</strong> <span class="side-note">if needed</span><br>↓<br><strong>Summarizer</strong><br>↓<br><strong>Verifier</strong><br>↓<br><strong>Grounded answer</strong></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="eyebrow">Research principle</div><h3>Why insufficient evidence?</h3>', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-note">When the retrieved text cannot support a defensible claim, MedAssist says so explicitly. It treats uncertainty as an answer state instead of filling the gap with a guess.</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="hero"><div class="eyebrow">Biomedical intelligence / grounded by design</div><h1>MedAssist</h1><p>Evidence-Grounded Biomedical Question Answering</p><span class="architecture">Agentic RAG&nbsp; • &nbsp;Hybrid Retrieval&nbsp; • &nbsp;Multi-Agent Verification</span></div>',
    unsafe_allow_html=True,
)

if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]
if "example_clicked" in st.session_state:
    st.session_state.question = st.session_state.pop("example_clicked")

st.markdown('<div class="section-label">Ask a biomedical question</div>', unsafe_allow_html=True)
question = st.text_area(
    "Biomedical question",
    key="question",
    height=105,
    label_visibility="collapsed",
    placeholder="Ask about a treatment, mechanism, diagnosis, or outcome...",
)
example_columns = st.columns(4)
for column, example in zip(example_columns, EXAMPLES):
    if column.button(example, key=f"example_{example}", use_container_width=True):
        st.session_state.example_clicked = example
        st.rerun()

run = st.button("Run Analysis  →", type="primary", use_container_width=False)
if run:
    if not question.strip():
        st.warning("Enter a biomedical question first.")
    else:
        try:
            with st.status("Running the evidence pipeline...", expanded=True) as status:
                st.write("Hybrid retrieval is searching the pooled biomedical index.")
                st.write("The summarizer and independent verifier may take 20–30 seconds.")
                result = get_orchestrator().run(question.strip())
                status.update(label="Analysis complete", state="complete", expanded=False)
            st.session_state.result = result
        except Exception as exc:
            message = str(exc)
            if "429" in message or "rate_limit" in message.lower() or "RateLimit" in message:
                st.error("Groq rate limit reached. Please wait for the quota reset and try again.")
            else:
                st.error(f"Analysis could not be completed: {type(exc).__name__}: {message}")


if "result" in st.session_state:
    result = st.session_state.result
    answer = answer_class(str(result.get("answer", "insufficient_evidence")))
    answer_label = answer.replace("_", " ")
    evidence = result.get("evidence_used") or []
    fallback = bool(result.get("used_web_fallback"))
    st.markdown('<div class="section-label">Analysis result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-card {answer}"><div class="answer-kicker">Verified answer</div><div class="answer-value">{html.escape(answer_label)}</div><div class="rationale">{html.escape(str(result.get("rationale") or "No rationale returned."))}</div></div>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(3)
    metric_columns[0].markdown(f'<div class="metric"><div class="metric-label">Confidence heuristic</div><div class="metric-value">◉ {html.escape(str(result.get("confidence_level", "unknown")).title())}</div></div>', unsafe_allow_html=True)
    fallback_label = "Used web fallback" if fallback else "Local corpus only"
    metric_columns[1].markdown(f'<div class="metric"><div class="metric-label">Evidence route</div><div class="metric-value">{html.escape(fallback_label)}</div></div>', unsafe_allow_html=True)
    metric_columns[2].markdown(f'<div class="metric"><div class="metric-label">Sources used</div><div class="metric-value">{len(evidence)}</div></div>', unsafe_allow_html=True)
    with st.expander("View Evidence Sources", expanded=True):
        render_evidence(evidence)