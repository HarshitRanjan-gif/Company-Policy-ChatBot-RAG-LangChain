import streamlit as st
import time
import uuid

from rag import ask_rag


# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📄",
    layout="wide"
)


# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("📄 Company Policy Assistant")

st.sidebar.markdown("---")

st.sidebar.write("### Framework")
st.sidebar.success("LangChain (Conversational RAG)")

st.sidebar.write("### Vector Database")
st.sidebar.success("FAISS")

st.sidebar.write("### Embedding Model")
st.sidebar.success("BAAI / BGE-M3")

st.sidebar.write("### LLM")
st.sidebar.success("Groq - Llama 3.1 8B")

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.session_state.session_id = str(uuid.uuid4())

    st.rerun()


# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.title("📄 Company Policy Assistant")

st.write(
    "Ask any question related to the company HR policies."
)

st.markdown("---")


# ----------------------------------------------------
# Session State
# ----------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# ----------------------------------------------------
# Display Previous Messages
# ----------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            # Response Time
            st.caption(
                f"⏱ Response Time: {message['time']} sec"
            )

            # Source Pages
            st.write("### 📄 Source Pages")

            st.write(
                ", ".join(
                    f"Page {page}" for page in message["pages"]
                )
            )

            # Retrieved Context
            with st.expander("📚 Retrieved Context"):

                st.write(message["context"])


# ----------------------------------------------------
# Chat Input
# ----------------------------------------------------

question = st.chat_input(
    "Ask your question..."
)


# ----------------------------------------------------
# User Question
# ----------------------------------------------------

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching company policies..."):

            start = time.time()

            try:

                answer, context, pages = ask_rag(
                    question,
                    st.session_state.session_id
                )

            except Exception as e:

                error = str(e)

                if "rate limit" in error.lower():

                    answer = (
                        "⚠️ **Groq Rate Limit Reached**\n\n"
                        "Please wait for about **20-30 seconds** and try again."
                    )

                elif "connection" in error.lower():

                    answer = (
                        "⚠️ **Unable to connect to the Groq API.**\n\n"
                        "Please check your internet connection or try again in a few seconds."
                    )

                else:

                    answer = (
                        f"⚠️ **Unexpected Error**\n\n"
                        f"{error}"
                    )

                context = ""
                pages = []

            end = time.time()

        elapsed = round(end - start, 2)

        # Display Answer
        st.markdown(answer)

        # Response Time
        st.caption(
            f"⏱ Response Time: {elapsed} sec"
        )

        # Source Pages
        if pages:

            st.write("### 📄 Source Pages")

            st.write(
                ", ".join(
                    f"Page {page}" for page in pages
                )
            )

        # Retrieved Context
        if context:

            with st.expander("📚 Retrieved Context"):

                st.write(context)

    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer,
            "time": elapsed,
            "context": context,
            "pages": pages
        }

    )