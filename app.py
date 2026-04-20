# ============================================================
# app.py
#
# Streamlit chat interface for the Feature Selection Bot.
#
# Drop this file at the project root alongside main.py.
# Run with:  python -m streamlit run app.py
#
# v2 improvements over v1:
#   - Color-coded sidebar metrics (green/red/grey numbers)
#   - Colored coverage progress bar (green fill)
#   - Custom avatars for user (👤) and bot (🔬)
#   - Dataset info strip at top of chat
#   - Typing indicator while agent is thinking
#   - Timestamps on every message
#   - "New Session" button in sidebar to re-upload without
#     refreshing the browser
#
# Zero changes to any existing file. Additive only.
# ============================================================

import glob
import os
import tempfile
from datetime import datetime

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="FT Bot — Feature Selection Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar padding */
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

    /* ── Colored metric numbers ── */
    .metric-block        { text-align: center; padding: 10px 4px; border-radius: 8px; }
    .metric-label        { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                           letter-spacing: 0.05em; color: #888; margin-bottom: 4px; }
    .metric-value        { font-size: 2rem; font-weight: 800; line-height: 1; }
    .metric-value.green  { color: #2ecc71; }
    .metric-value.red    { color: #e74c3c; }
    .metric-value.grey   { color: #95a5a6; }
    .metric-value.white  { color: #ecf0f1; }

    /* ── Coverage bar ── */
    .coverage-wrap       { margin: 6px 0 14px 0; }
    .coverage-label      { font-size: 0.78rem; color: #aaa; margin-bottom: 5px; }
    .coverage-bar-bg     { background: #2c3e50; border-radius: 6px; height: 10px; width: 100%; }
    .coverage-bar-fill   { background: linear-gradient(90deg, #27ae60, #2ecc71);
                           border-radius: 6px; height: 10px; transition: width 0.4s ease; }

    /* ── Draft mode badge ── */
    .badge-draft         { display:inline-block; background:#d4a017; color:#000;
                           padding:3px 10px; border-radius:4px; font-size:0.78rem;
                           font-weight:700; margin-bottom:6px; }

    /* ── Dataset info strip ── */
    .dataset-strip       { background: #1a2332; border-left: 3px solid #3498db;
                           border-radius: 0 6px 6px 0; padding: 8px 14px;
                           margin-bottom: 16px; font-size: 0.82rem; color: #a8c6e8; }
    .dataset-strip span  { color: #ecf0f1; font-weight: 600; }

    /* ── Message timestamp ── */
    .msg-timestamp       { font-size: 0.68rem; color: #555; margin-top: 3px; }

    /* ── Typing indicator ── */
    .typing-indicator    { display: flex; align-items: center; gap: 5px;
                           padding: 10px 0; color: #888; font-size: 0.82rem; }
    .typing-dot          { width: 7px; height: 7px; border-radius: 50%;
                           background: #3498db; animation: blink 1.2s infinite; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes blink     { 0%,80%,100% { opacity:0.2; } 40% { opacity:1; } }

    /* ── Chat message spacing ── */
    .stChatMessage       { margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────

def _init_session_state():
    defaults = {
        "data_loaded"      : False,
        "agent"            : None,
        "state"            : None,
        # Each entry: {"role", "content", "timestamp"}
        "chat_history"     : [],
        "output_dir"       : None,
        "last_report_path" : None,
        "pipeline_error"   : None,
        "total_features"   : 0,
        "train_filename"   : "",
        "labels_filename"  : "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session_state()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _now_str() -> str:
    return datetime.now().strftime("%H:%M")


def _colored_metric(label: str, value, color: str) -> str:
    return (
        f'<div class="metric-block">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {color}">{value}</div>'
        f'</div>'
    )


def _coverage_bar(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div class="coverage-wrap">'
        f'<div class="coverage-label">Coverage &nbsp;'
        f'<strong style="color:#ecf0f1">{pct}%</strong></div>'
        f'<div class="coverage-bar-bg">'
        f'<div class="coverage-bar-fill" style="width:{pct}%"></div>'
        f'</div></div>'
    )


# ─────────────────────────────────────────────────────────────
# Pipeline bootstrap
# ─────────────────────────────────────────────────────────────

def _bootstrap_pipeline(train_bytes: bytes, labels_bytes: bytes,
                         train_name: str, labels_name: str):
    # Use a fixed temp dir for CSVs (loader.py needs paths on disk)
    # but save reports to ./outputs/ so they're always findable
    tmp_dir     = tempfile.mkdtemp(prefix="ftbot_")
    train_path  = os.path.join(tmp_dir, "train.csv")
    labels_path = os.path.join(tmp_dir, "labels.csv")

    # Fixed output directory — lives in the project folder, survives the session
    output_dir = os.path.abspath(os.path.join(os.getcwd(), "outputs"))
    os.makedirs(output_dir, exist_ok=True)

    with open(train_path,  "wb") as f:
        f.write(train_bytes)
    with open(labels_path, "wb") as f:
        f.write(labels_bytes)

    from pipeline.session import build_session
    session = build_session(train_path, labels_path, output_dir)
    session["output_dir"] = output_dir

    from graph.graph import build_graph
    agent, state = build_graph(session)
    state.setdefault("overview_mode", None)

    st.session_state.agent            = agent
    st.session_state.state            = state
    st.session_state.output_dir       = output_dir
    st.session_state.total_features   = len(session.get("feature_cols", []))
    st.session_state.data_loaded      = True
    st.session_state.chat_history     = []
    st.session_state.pipeline_error   = None
    st.session_state.train_filename   = train_name
    st.session_state.labels_filename  = labels_name
    st.session_state.last_report_path = None


# ─────────────────────────────────────────────────────────────
# Agent turn
# ─────────────────────────────────────────────────────────────

def _run_turn(user_text: str):
    state = st.session_state.state
    state["messages"].append({"role": "user", "content": user_text})

    state = st.session_state.agent.invoke(state)
    st.session_state.state = state

    ts        = _now_str()
    is_report = (state.get("intent") == "REPORT")

    st.session_state.chat_history.append(
        {"role": "user", "content": user_text, "timestamp": ts}
    )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": state.get("last_response", ""),
         "timestamp": ts, "is_report": is_report}
    )

    if is_report:
        _cache_latest_report()



def _cache_latest_report():
    output_dir = st.session_state.output_dir
    if not output_dir:
        return
    pattern = os.path.join(output_dir, "ft_selection_report_*.xlsx")
    matches = sorted(glob.glob(pattern))
    if matches:
        st.session_state.last_report_path = matches[-1]


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.markdown("## 🔬 FT Bot")
        st.caption("Feature Selection Assistant")
        st.divider()

        # ── New Session button ────────────────────────────────
        if st.session_state.data_loaded:
            if st.button("🔄 New Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
            st.divider()

        if not st.session_state.data_loaded:
            st.info("Upload your datasets to begin.")
            return

        state      = st.session_state.state
        decisions  = state.get("decisions", {})
        total      = st.session_state.total_features
        kept       = sum(1 for v in decisions.values() if v == "keep")
        dropped    = sum(1 for v in decisions.values() if v == "drop")
        pending    = total - kept - dropped
        cov_pct    = round((kept + dropped) / total * 100, 1) if total else 0.0
        draft_mode = state.get("draft_mode", False)

        # Draft mode badge
        if draft_mode:
            st.markdown(
                '<div class="badge-draft">⚠ Draft mode active</div>',
                unsafe_allow_html=True,
            )
            st.caption("Auto-decisions loaded. Review and confirm.")
            st.divider()

        # ── Session Summary ───────────────────────────────────
        st.markdown("### Session Summary")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(_colored_metric("Total",   total,   "white"), unsafe_allow_html=True)
        with c2:
            st.markdown(_colored_metric("Kept",    kept,    "green"), unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown(_colored_metric("Pending", pending, "grey"),  unsafe_allow_html=True)
        with c4:
            st.markdown(_colored_metric("Dropped", dropped, "red"),   unsafe_allow_html=True)

        st.markdown(_coverage_bar(cov_pct), unsafe_allow_html=True)
        st.divider()

        # ── Export ────────────────────────────────────────────
        st.markdown("### Export")

        if st.button("📥 Export Report", use_container_width=True):
            with st.spinner("Generating report..."):
                _run_turn("generate the report")
            st.rerun()

        if st.session_state.last_report_path and \
                os.path.exists(st.session_state.last_report_path):
            with open(st.session_state.last_report_path, "rb") as f:
                report_bytes = f.read()
            fname = os.path.basename(st.session_state.last_report_path)
            st.download_button(
                label="⬇ Download Latest Report",
                data=report_bytes,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.divider()
        st.caption("Type anything in the chat — no buttons needed.")


# ─────────────────────────────────────────────────────────────
# Upload screen
# ─────────────────────────────────────────────────────────────

def _render_upload_screen():
    st.title("Feature Selection Bot")
    st.markdown(
        "Upload your dataset files below to begin. "
        "The pipeline will run automatically — **first load takes 1–2 minutes**."
    )
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader(
            "train.csv — feature columns",
            type=["csv"],
            key="upload_train",
        )
    with col2:
        labels_file = st.file_uploader(
            "labels.csv — binary churn target (0 / 1)",
            type=["csv"],
            key="upload_labels",
        )

    if train_file and labels_file:
        st.info("Both files received. Running statistical pipeline...")
        with st.spinner("Running pipeline — statistical tests across all features. Please wait..."):
            try:
                _bootstrap_pipeline(
                    train_file.read(), labels_file.read(),
                    train_file.name,   labels_file.name,
                )
                st.success("Pipeline complete. Starting chat...")
                st.rerun()
            except FileNotFoundError as e:
                st.error(f"File error: {e}")
                st.session_state.pipeline_error = str(e)
            except ValueError as e:
                st.error(
                    f"Validation error: {e}\n\nCheck that your labels file is "
                    "binary (0/1) and has the same number of rows as your train file."
                )
                st.session_state.pipeline_error = str(e)
            except Exception as e:
                st.error(f"Unexpected pipeline error: {e}")
                st.session_state.pipeline_error = str(e)

    elif train_file or labels_file:
        st.warning("Please upload **both** files to continue.")

    if st.session_state.pipeline_error and not (train_file or labels_file):
        st.error(f"Previous error: {st.session_state.pipeline_error}")
        if st.button("Clear error and retry"):
            st.session_state.pipeline_error = None
            st.rerun()


# ─────────────────────────────────────────────────────────────
# Chat screen
# ─────────────────────────────────────────────────────────────

def _render_chat_screen():

    # ── Dataset info strip ────────────────────────────────────
    train_name  = st.session_state.train_filename  or "train.csv"
    labels_name = st.session_state.labels_filename or "labels.csv"
    total       = st.session_state.total_features
    st.markdown(
        f'<div class="dataset-strip">'
        f'<span>{total} features</span> &nbsp;·&nbsp; '
        f'<span>{train_name}</span> &nbsp;+&nbsp; '
        f'<span>{labels_name}</span> &nbsp;loaded'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chat history ──────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            "_No messages yet. Ask anything — try:_\n\n"
            "- `give me an overview of the dataset`\n"
            "- `what do you think about the null-heavy columns?`\n"
            "- `analyse Var126`\n"
            "- `what's the status so far?`"
        )
    else:
        for i, msg in enumerate(st.session_state.chat_history):
            role      = msg["role"]
            content   = msg["content"]
            timestamp = msg.get("timestamp", "")
            avatar    = "👤" if role == "user" else "🔬"

            with st.chat_message(role, avatar=avatar):
                st.markdown(content)
                if timestamp:
                    st.markdown(
                        f'<div class="msg-timestamp">{timestamp}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Inline download button after report messages ──
                # Shown on the last assistant message when a report exists.
                # This replaces the confusing "link provided" text with
                # an actual button the user can see right in the chat.
                is_last_msg   = (i == len(st.session_state.chat_history) - 1)
                is_report_msg = msg.get("is_report", False)
                report_path   = st.session_state.last_report_path

                if (role == "assistant" and is_report_msg and
                        report_path and os.path.exists(report_path)):
                    fname = os.path.basename(report_path)
                    with open(report_path, "rb") as f:
                        report_bytes = f.read()
                    st.markdown("---")
                    st.success(f"✅ Report saved to `outputs/{fname}`")
                    st.download_button(
                        label="⬇️ Download Report",
                        data=report_bytes,
                        file_name=fname,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"inline_dl_{i}",
                    )

    # ── Typing indicator placeholder ──────────────────────────
    thinking_placeholder = st.empty()

    # ── Input ─────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about the dataset...")

    if user_input:
        thinking_placeholder.markdown(
            '<div class="typing-indicator">'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '&nbsp; FT Bot is thinking...</div>',
            unsafe_allow_html=True,
        )
        try:
            _run_turn(user_input.strip())
        except Exception as e:
            st.session_state.chat_history.append({
                "role"     : "assistant",
                "content"  : f"⚠ Agent error: {e}",
                "timestamp": _now_str(),
            })
        finally:
            thinking_placeholder.empty()

        st.rerun()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

_render_sidebar()

if not st.session_state.data_loaded:
    _render_upload_screen()
else:
    _render_chat_screen()