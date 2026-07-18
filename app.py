"""
Hackathon demo skeleton.
Generic Streamlit app: takes input -> runs core logic -> displays output.
When the track is revealed, edit ONLY the `run_core_logic` function below.
Everything else (UI, API wiring, RAG helpers) already works.
"""

import streamlit as st
from openai import OpenAI

from rag import build_index, retrieve

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Demo", page_icon="⚡", layout="wide")

# API key: pulled from Streamlit secrets in the cloud, or sidebar as fallback.
def get_client():
    key = st.secrets.get("OPENAI_API_KEY", None)
    if not key:
        key = st.session_state.get("api_key_input", None)
    if not key:
        return None
    return OpenAI(api_key=key)


# ----------------------------------------------------------------------------
# CORE LOGIC  ---  THIS IS THE ONLY PART YOU SWAP WHEN THE TRACK DROPS
# ----------------------------------------------------------------------------
def run_core_logic(user_input: str, client: OpenAI, context_docs: list[str]) -> str:
    """
    Right now: a simple RAG answer.
    Retrieve relevant context, stuff it into the prompt, generate an answer.
    Replace the body with whatever the challenge needs — a classifier,
    a hybrid ML+domain model, an agent loop, etc. The signature can stay.
    """
    context = "\n\n".join(context_docs) if context_docs else "(no context)"

    system = (
        "You are a helpful assistant. Use the provided context when relevant. "
        "If the context does not contain the answer, say so plainly."
    )
    prompt = f"Context:\n{context}\n\nQuestion: {user_input}"

    resp = client.chat.completions.create(
        model=st.session_state.get("model", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.session_state["model"] = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"], index=0
    )
    st.text_input(
        "OpenAI API key (fallback)",
        type="password",
        key="api_key_input",
        help="Only needed if not set in Streamlit secrets.",
    )
    st.caption("Set OPENAI_API_KEY in app secrets for the deployed version.")

    st.divider()
    st.subheader("Knowledge base")
    uploaded = st.file_uploader(
        "Upload text/markdown docs (optional)",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

# ----------------------------------------------------------------------------
# Build (or rebuild) the RAG index from uploads
# ----------------------------------------------------------------------------
if "index" not in st.session_state:
    st.session_state["index"] = None

if uploaded:
    docs = [f.read().decode("utf-8", errors="ignore") for f in uploaded]
    client = get_client()
    if client:
        with st.spinner("Indexing documents..."):
            st.session_state["index"] = build_index(docs, client)
        st.sidebar.success(f"Indexed {len(docs)} document(s).")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("⚡ Demo")
st.write("Swap the placeholder logic when the challenge is revealed.")

user_input = st.text_area("Your input", height=120, placeholder="Ask something...")

if st.button("Run", type="primary"):
    client = get_client()
    if not client:
        st.error("No API key. Add it in the sidebar or in app secrets.")
    elif not user_input.strip():
        st.warning("Enter some input first.")
    else:
        # Retrieve context if we have an index
        context_docs = []
        if st.session_state["index"] is not None:
            context_docs = retrieve(user_input, st.session_state["index"], client, k=3)

        with st.spinner("Working..."):
            try:
                output = run_core_logic(user_input, client, context_docs)
                st.subheader("Output")
                st.write(output)
                if context_docs:
                    with st.expander("Retrieved context"):
                        for i, d in enumerate(context_docs, 1):
                            st.markdown(f"**Chunk {i}**")
                            st.caption(d[:500] + ("..." if len(d) > 500 else ""))
            except Exception as e:
                st.error(f"Something broke: {e}")
