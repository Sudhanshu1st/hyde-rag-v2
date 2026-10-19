"""Streamlit demo: ask the food-safety corpus and inspect every step.

Run:  uv run streamlit run app/streamlit_app.py
Needs: the food corpus (download + ingest-food) and its index (index-food), plus a free Groq key
in .env for HyDE and answers.
"""

from __future__ import annotations

import re

import streamlit as st
from dotenv import load_dotenv

from hyde_rag.pipeline import MODES, PipelineConfig, RAGPipeline, load_pipeline_config

load_dotenv()
st.set_page_config(page_title="HyDE RAG v2", layout="wide")
st.title("HyDE RAG v2 · food-safety Q&A")
st.caption(
    "Answers grounded in public FDA, USDA FSIS, Codex and FSSAI documents. Every sentence cites its source."
)


@st.cache_resource(show_spinner="Loading the corpus and models...")
def get_pipeline(mode: str, n_passages: int) -> RAGPipeline:
    base = load_pipeline_config()
    return RAGPipeline(PipelineConfig(**{**base.__dict__, "mode": mode, "n_passages": n_passages}))


base = load_pipeline_config()
with st.sidebar:
    st.header("Retrieval")
    mode = st.selectbox(
        "Mode",
        MODES,
        index=MODES.index(base.mode),
        help="hyde = embed hypothetical passages written by an LLM, then search",
    )
    n_passages = st.slider("Hypothetical passages (N)", 1, 4, base.n_passages, disabled="hyde" not in mode)
    st.markdown("Free-tier LLMs via Groq. Nothing here costs money.")

question = st.text_input("Question", placeholder="How fast must cooked food be cooled?")
if st.button("Ask", type="primary") and question.strip():
    try:
        res = get_pipeline(mode, n_passages).query(question.strip())
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    if res.abstained:
        st.warning(res.answer + (" (retrieval gate)" if res.gated else ""))
    else:
        st.subheader("Answer")
        st.markdown(re.sub(r"\[([^\]]+)\]", r"`[\1]`", res.answer))
        if res.invalid_citations or res.uncited_sentences:
            st.error(
                f"Citation check failed: unknown IDs {res.invalid_citations}, "
                f"uncited sentences {res.uncited_sentences}"
            )
        else:
            st.success("Citation check passed: every sentence cites a retrieved passage.")

    cited = {cid for c in res.citations for cid in c["cited_ids"]}
    col1, col2 = st.columns(2)
    with col1, st.expander(f"Hypothetical documents ({len(res.hypothetical_documents)})", expanded=True):
        if not res.hypothetical_documents:
            st.write("Not used in this mode.")
        for i, p in enumerate(res.hypothetical_documents, 1):
            st.markdown(f"**#{i}** {p}")
    with col2, st.expander("Timings and tokens", expanded=True):
        st.json({"timings_ms": res.timings_ms, "llm_tokens": res.llm_tokens})
    st.subheader("Retrieved passages")
    for r in res.retrieved:
        mark = "✅ cited" if r["chunk_id"] in cited else ""
        with st.expander(f"[{r['chunk_id']}] {r['title']} · {r['section'][:80]} {mark}"):
            st.write(r["text"])
