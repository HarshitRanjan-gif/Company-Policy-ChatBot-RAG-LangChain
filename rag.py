import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)


# -----------------------------------------
# Load Environment Variables
# -----------------------------------------

load_dotenv()


# -----------------------------------------
# Embedding Model
# -----------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)


# -----------------------------------------
# Load FAISS Vector Store
# -----------------------------------------

vector_db = FAISS.load_local(
    "vector_store",
    embedding_model,
    allow_dangerous_deserialization=True
)


# -----------------------------------------
# Retriever
# -----------------------------------------

retriever = vector_db.as_retriever(
    search_kwargs={
        "k": 3
    }
)

# -----------------------------------------
# Groq LLM
# -----------------------------------------

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.1-8b-instant",

    temperature=0.2,

    max_tokens=500

)


# =========================================
# History Aware Prompt
# =========================================

contextualize_q_prompt = ChatPromptTemplate.from_messages(

    [

        (
            "system",

            "Given a chat history and the latest user question "
            "which might reference previous context, "
            "formulate a standalone question. "
            "Do NOT answer it."

        ),

        MessagesPlaceholder("chat_history"),

        ("human", "{input}")

    ]

)


# =========================================
# History Aware Retriever
# =========================================

history_aware_retriever = create_history_aware_retriever(

    llm,

    retriever,

    contextualize_q_prompt

)


# =========================================
# QA Prompt
# =========================================

qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an AI HR Policy Assistant.

Use ONLY the following retrieved context to answer the user's question.

If the answer is not present in the context, say:

"I couldn't find this information in the company policies."

--------------------
Retrieved Context:
{context}
--------------------

Provide a detailed answer.
"""
        ),

        MessagesPlaceholder("chat_history"),

        ("human", "{input}")
    ]
)


# =========================================
# Stuff Documents Chain
# =========================================

question_answer_chain = create_stuff_documents_chain(

    llm,

    qa_prompt

)


# =========================================
# Retrieval Chain
# =========================================

rag_chain = create_retrieval_chain(

    history_aware_retriever,

    question_answer_chain

)


# =========================================
# Chat History Storage
# =========================================

store = {}


def get_session_history(
    session_id: str
) -> BaseChatMessageHistory:

    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]

# =========================================
# Runnable With History
# =========================================

conversational_rag = RunnableWithMessageHistory(

    rag_chain,

    get_session_history,

    input_messages_key="input",

    history_messages_key="chat_history",

    output_messages_key="answer"

)


# =========================================
# Main Function
# =========================================

def ask_rag(question, session_id):

    response = conversational_rag.invoke(

        {
            "input": question
        },

        config={
            "configurable": {
                "session_id": session_id
            }
        }

    )

    context = ""

    pages = []

    for doc in response["context"]:

        context += doc.page_content
        context += "\n\n"

        page = doc.metadata.get("page", None)

        if page is not None:
            pages.append(page + 1)

    pages = sorted(set(pages))

    return response["answer"], context, pages