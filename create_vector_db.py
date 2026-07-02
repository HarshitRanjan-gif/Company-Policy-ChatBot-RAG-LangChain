from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


loader = PyPDFLoader(
    "data/company_policy.pdf"
)

documents = loader.load()

print("Pages:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(

    chunk_size=1000,

    chunk_overlap=200

)

chunks = text_splitter.split_documents(
    documents
)

print("Chunks:", len(chunks))


embedding_model = HuggingFaceEmbeddings(

    model_name="BAAI/bge-m3"
)


vector_db = FAISS.from_documents(

    chunks,

    embedding_model

)


vector_db.save_local(
    "vector_store"
)

print("Vector Database Created Successfully!")