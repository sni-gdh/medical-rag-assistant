from langchain_community.vectorstores import Chroma
from config import CHROMA_DIR

def create_vectorstore(chunks, embedding):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_DIR
    )
    db.persist()
    return db

def load_vectorstore(embedding):
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )