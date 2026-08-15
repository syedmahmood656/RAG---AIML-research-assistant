# app.py
# Streamlit UI for the ArXiv RAG backend.
#
# Flow:
#   1. User types a question, hits Enter.
#   2. We POST /query to the FastAPI backend with stream=True.
#   3. Parse SSE: a `sources` event arrives first (chunks to cite),
#      then a sequence of `token` events that we append to a single
#      placeholder as they arrive (true progressive rendering).
#   4. The final answer + source list are stored in st.session_state
#      so the user can scroll back through prior questions.

import json
import os
import re

import requests
import streamlit as st

# Configurable via env when running in Docker. Defaults to localhost
# so a bare `streamlit run` against a locally-running uvicorn also works.
API_URL = os.environ.get("API_URL", "http://localhost:7860")

st.set_page_config(
    page_title="ArXiv RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ArXiv RAG — Research-Paper Q&A")
st.caption(
    "Ask questions about ~150 recent cs.AI / cs.LG papers on ArXiv. "
    "Answers are grounded in retrieved chunks; the sources used are shown below."
)


# ---------- helpers ----------

_SSE_LINE_RE = re.compile(r"^(?P<event>event|data):\s?(?P<value>.*)$")


def _parse_sse_event(block: str):
    """
    Parse one SSE message block (one or more lines, blank-line terminated).
    Returns (event_name, data_string) or (None, None) if incomplete/empty.
    """
    event = None
    data_lines = []
    for line in block.splitlines():
        m = _SSE_LINE_RE.match(line)
        if not m:
            continue
        if m.group("event") == "event":
            event = m.group("value").strip()
        elif m.group("event") == "data":
            data_lines.append(m.group("value"))

    if event is None and not data_lines:
        return None, None
    return event, "\n".join(data_lines)


def stream_query(question: str, k: int):
    """
    Generator that yields ('sources', [...]) once, then many ('token', str),
    then ('done', None) at the end. Raises on network/HTTP errors.
    """
    resp = requests.post(
        f"{API_URL}/query",
        json={"question": question, "k": k},
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    # SSE messages are separated by a blank line. We accumulate bytes
    # into `buf` until we see one, then hand the block to the parser.
    buf = ""
    for raw in resp.iter_content(chunk_size=1, decode_unicode=True):
        if raw is None:
            continue
        buf += raw
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            event, data = _parse_sse_event(block)
            if event is None:
                continue
            if event == "sources":
                payload = json.loads(data)
                yield ("sources", payload.get("sources", []))
            elif event == "token":
                yield ("token", data)
            elif event == "done":
                return
            elif event == "error":
                err = json.loads(data)
                raise RuntimeError(err.get("message", "unknown backend error"))


def render_sources(sources):
    """Render the retrieved-chunks panel below the answer."""
    if not sources:
        return
    st.markdown("### 📄 Sources")
    for src in sources:
        title = src.get("title", "Untitled")
        # Build a nice header line with title + arxiv link if we have one.
        pdf = src.get("pdf_url", "")
        if pdf:
            entry_id = src.get("entry_id") or pdf
            arxiv_id = entry_id.rsplit("/", 1)[-1]
            header = f"**[{title}]({entry_id})** — `{arxiv_id}`"
        else:
            header = f"**{title}**"
        with st.expander(header, expanded=False):
            authors = src.get("authors", "Unknown")
            published = src.get("published", "")
            category = src.get("category", "")
            bits = [f"**Authors:** {authors}"]
            if published:
                bits.append(f"**Published:** {published}")
            if category:
                bits.append(f"**Category:** `{category}`")
            st.markdown("  \n".join(bits))
            st.markdown("---")
            st.markdown(src.get("snippet", ""))


# ---------- session state ----------

if "history" not in st.session_state:
    # list of {"question": str, "answer": str, "sources": list}
    st.session_state.history = []


# ---------- sidebar ----------

with st.sidebar:
    st.header("Settings")
    k = st.slider("Retrieved chunks (k)", min_value=1, max_value=10, value=4, step=1)
    st.caption(
        f"Backend: `{API_URL}`"
    )
    if st.button("� Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------- form ----------

with st.form("query_form", clear_on_submit=True):
    question = st.text_area(
        "Your question",
        placeholder="e.g. What are the computational challenges of Seiberg Dualities?",
        height=100,
    )
    submitted = st.form_submit_button("Ask", use_container_width=True)


# ---------- run ----------

if submitted and question.strip():
    # Render the question immediately so it stays visible while the
    # answer streams in below.
    st.markdown(f"#### ❓ {question}")

    answer_placeholder = st.empty()
    sources_panel = st.container()

    full_answer = ""
    sources = []

    try:
        for kind, payload in stream_query(question.strip(), k):
            if kind == "sources":
                sources = payload
                # Show sources as soon as we have them, even before the
                # first token arrives from the LLM.
                with sources_panel:
                    render_sources(sources)
            elif kind == "token":
                full_answer += payload
                # Use markdown so basic formatting (lists, code) renders.
                answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)
    except requests.exceptions.RequestException as e:
        answer_placeholder.error(f"**Backend unreachable:** {e}\n\nIs the API running at `{API_URL}`?")
        full_answer = ""
    except RuntimeError as e:
        answer_placeholder.error(f"**Backend error:** {e}")
        full_answer = ""
    except Exception as e:
        answer_placeholder.error(f"**Unexpected error:** {e}")
        full_answer = ""

    # Persist to history (skip empty/failed answers).
    if full_answer.strip():
        st.session_state.history.insert(
            0, {"question": question, "answer": full_answer, "sources": sources}
        )


# ---------- history (rendered below the latest answer) ----------

if st.session_state.history:
    with st.expander(f"� Previous questions ({len(st.session_state.history) - 1 if len(st.session_state.history) > 1 else 0})", expanded=False):
        for i, item in enumerate(st.session_state.history[1:], start=1):
            st.markdown(f"---\n**Q{i}. {item['question']}**")
            st.markdown(item["answer"])
            render_sources(item.get("sources", []))
