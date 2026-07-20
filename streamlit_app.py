"""
Streamlit chat interface for the AI Customer Support Agent.

Run with:
    streamlit run streamlit_app.py

Requires OPENAI_API_KEY and PINECONE_API_KEY to be set (see .env.example)
and an already-populated Pinecone index (see README for the ingestion
pipeline that feeds it).
"""

from __future__ import annotations

import streamlit as st

from app.service import ask
from app.vector_store import index_stats

st.set_page_config(
    page_title="Ornativa Customer Support",
    page_icon="💎",
    layout="centered",
)

st.title("💎 Ornativa Customer Support Agent")
st.caption("Ask about products, warranty, shipping, returns or jewellery care.")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Session")

    if st.button("🔄 New conversation"):
        st.session_state.conversation_history = []
        st.session_state.messages = []
        st.rerun()

    with st.expander("Vector store status"):
        try:
            st.json(index_stats())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not reach Pinecone: {exc}")

# Replay chat history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Type your question...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(user_question, st.session_state.conversation_history)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
            else:
                st.markdown(result["final_response"])

                if result["sources"]:
                    with st.expander("Sources"):
                        for doc in result["sources"]:
                            st.markdown(f"**{doc.metadata.get('document_name', 'Unknown')}**")
                            st.caption(doc.page_content[:300])

                st.session_state.conversation_history = result["conversation_history"]
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["final_response"]}
                )
