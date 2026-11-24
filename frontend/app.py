# Streamlit UI for the ArXiv RAG backend.
#
#   user question -> POST /query (SSE) -> `sources` first (cards), then `token`s streamed into
#   the chat bubble. Full history is kept in st.session_state and re-rendered on every run.

import html
import os

import requests
import streamlit as st

from sse_client import BackendError, fetch_info, stream_query

API_URL = os.environ.get("API_URL", "http://localhost:7860")

st.set_page_config(page_title="ArXiv RAG", page_icon="🧠", layout="wide")

EXAMPLES = [
    ("🔭", "Physics meets ML", "How can machine learning be used to trace Seiberg dualities?"),
    ("🤖", "Robotics", "How do humanoid robots learn whole-body safety with control barrier functions?"),
    ("⚡", "Efficient AI", "How can sponge attacks increase inference energy in spiking neural networks?"),
    ("🧪", "Science tools", "What infrastructure helps synthesize chemistry literature from claims?"),
]

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 980px;}
.hero {text-align: center; padding: 1.6rem 0 0.6rem;}
.hero h1 {font-size: 2.6rem; font-weight: 800; margin: 0;
  background: linear-gradient(90deg, #8b9bff 0%, #5eead4 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero p {color: #9aa6c4; font-size: 1.05rem; margin: .5rem 0 0;}
.pill {display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .78rem;
  font-weight: 600; margin: 0 .3rem .3rem 0; border: 1px solid #2a3558; background: #131a2e; color: #c6d0ea;}
.pill.ok {border-color: #1f6f5c; color: #5eead4;}
.pill.bad {border-color: #7a2d3b; color: #ff8aa1;}
.pill.warn {border-color: #7a6a2d; color: #ffd479;}
.src-card {border: 1px solid #25304f; border-left: 4px solid #8b9bff; background: #101730;
  border-radius: 10px; padding: .7rem .9rem; margin: .55rem 0 .15rem;}
.src-head {display: flex; align-items: center; gap: .6rem;}
.src-rank {background: #8b9bff; color: #0b1020; font-weight: 800; border-radius: 6px;
  min-width: 1.5rem; text-align: center; padding: 0 .35rem;}
.src-title {font-weight: 650; flex: 1; line-height: 1.25;}
.src-score {color: #5eead4; font-size: .8rem; font-weight: 700; white-space: nowrap;}
.src-meta {color: #8f9bb8; font-size: .8rem; margin-top: .3rem;}
.bar {height: 4px; border-radius: 4px; background: #1c2542; margin-top: .55rem; overflow: hidden;}
.bar > div {height: 100%; background: linear-gradient(90deg, #8b9bff, #5eead4);}
.footnote {color: #7f8bab; font-size: .78rem; margin-top: .4rem;}
div[data-testid="stButton"] button {border-radius: 12px;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- helpers


@st.cache_data(ttl=10, show_spinner=False)
def get_info(api_url: str):
    return fetch_info(api_url)


def esc(x) -> str:
    return html.escape(str(x or ""))


def render_sources(sources, key_prefix: str):
    if not sources:
        return
    st.markdown("##### 📄 Sources")
    for s in sources:
        pct = int(round(float(s.get("score", 0)) * 100))
        bits = [b for b in (s.get("authors"), s.get("published"), s.get("category")) if b]
        meta = " · ".join(esc(b) for b in bits)
        if len(meta) > 180:
            meta = meta[:180] + "…"
        st.markdown(
            f"""<div class="src-card">
  <div class="src-head"><span class="src-rank">{s.get('rank')}</span>
    <span class="src-title">{esc(s.get('title'))}</span>
    <span class="src-score">{pct}% match</span></div>
  {f'<div class="src-meta">{meta}</div>' if meta else ''}
  <div class="bar"><div style="width:{pct}%"></div></div>
</div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Read excerpt"):
            st.text(s.get("snippet", ""))
            link = s.get("entry_id") or s.get("pdf_url")
            if link:
                st.link_button("Open on arXiv ↗", link)


def render_meta(meta):
    if not meta:
        return
    secs = meta.get("elapsed_ms", 0) / 1000
    mode = "generated answer" if meta.get("mode") == "llm" else "retrieval-only"
    st.markdown(f'<div class="footnote">⏱ {secs:.1f}s · {mode}</div>', unsafe_allow_html=True)


def render_message(i: int, msg):
    avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            return
        if msg.get("notice"):
            st.warning(msg["notice"], icon="⚠️")
        st.markdown(msg["content"])
        render_meta(msg.get("meta"))
        render_sources(msg.get("sources"), f"h{i}")


# ----------------------------------------------------------------------------- state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)

# ----------------------------------------------------------------------------- sidebar
info = get_info(API_URL)
with st.sidebar:
    st.markdown("### 🧠 ArXiv RAG")
    st.caption("Ask questions about recent cs.AI / cs.LG papers.")

    if info is None:
        st.markdown('<span class="pill bad">● API offline</span>', unsafe_allow_html=True)
        st.caption(f"Cannot reach `{API_URL}`. Start the backend first (see guid.md, step 5).")
    else:
        st.markdown('<span class="pill ok">● API online</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="pill">{info.get("chunks", "?"):,} chunks</span>', unsafe_allow_html=True)
        if info.get("llm_configured"):
            st.markdown(f'<span class="pill ok">LLM · {esc(info.get("llm_model"))}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn">Retrieval-only (no HF token)</span>', unsafe_allow_html=True)
        if not info.get("has_paper_metadata", True):
            st.warning(
                "This index has no paper titles / authors / links. Rebuild it - see guid.md, "
                "section 'Rebuild the index'.",
                icon="🧱",
            )

    st.divider()
    k = st.slider("Excerpts to retrieve (k)", 1, 10, 4, help="More excerpts = more context, slower answers.")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"Backend: `{API_URL}`")

# ----------------------------------------------------------------------------- main
if not st.session_state.messages:
    st.markdown(
        """<div class="hero"><h1>Research-paper Q&amp;A</h1>
<p>Grounded answers with cited excerpts from ~150 recent arXiv papers.</p></div>""",
        unsafe_allow_html=True,
    )
    st.write("")
    cols = st.columns(2)
    for n, (icon, label, question) in enumerate(EXAMPLES):
        with cols[n % 2]:
            if st.button(f"{icon}  **{label}**\n\n{question}", key=f"ex{n}", use_container_width=True):
                st.session_state.pending = question
                st.rerun()

for i, m in enumerate(st.session_state.messages):
    render_message(i, m)

typed = st.chat_input("Ask about the indexed papers…")
prompt = st.session_state.pending or typed
st.session_state.pending = None

if prompt and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        notice_ph = st.empty()
        answer_ph = st.empty()
        src_box = st.container()
        answer_ph.markdown("_Searching the papers…_")

        answer, notice, sources, meta, failed = "", "", [], None, False
        try:
            for kind, payload in stream_query(API_URL, prompt, k):
                if kind == "sources":
                    sources = payload.get("sources", [])
                    with src_box:
                        render_sources(sources, "live")
                    answer_ph.markdown("_Writing the answer…_")
                elif kind == "notice":
                    notice = payload.get("message", "")
                    notice_ph.warning(notice, icon="⚠️")
                elif kind == "token":
                    answer += payload
                    answer_ph.markdown(answer + " ▌")
                elif kind == "done":
                    meta = payload
            answer_ph.markdown(answer)
            render_meta(meta)
        except requests.exceptions.ConnectionError:
            failed = True
            answer_ph.error(
                f"**Can't reach the API at `{API_URL}`.** Start it with:\n\n"
                "`python -m uvicorn app:app --app-dir backend --host 127.0.0.1 --port 7860`"
            )
        except requests.exceptions.RequestException as e:
            failed = True
            answer_ph.error(f"**Request failed:** {e}")
        except BackendError as e:
            failed = True
            answer_ph.error(f"**Backend error:** {e}")
        except Exception as e:  # noqa: BLE001
            failed = True
            answer_ph.error(f"**Unexpected error:** {e}")

    if failed or not answer.strip():
        st.session_state.messages.pop()  # drop the unanswered question so history stays clean
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "notice": notice, "meta": meta}
        )
