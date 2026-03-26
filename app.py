import streamlit as st
import os

from utils.loader import load_documents
from rag.ingest import chunk_docs
from rag.embed import get_embedding
from rag.vector_store import create_vectorstore, load_vectorstore
from rag.retriever import get_retriever
from rag.pipeline import load_llm, generate_answer

st.title("🧬 Medical RAG Assistant")

@st.cache_resource
def setup():
    embedding = get_embedding()

    if os.path.exists("chroma_db"):
        db = load_vectorstore(embedding)
    else:
        docs = load_documents("data/docs")
        chunks = chunk_docs(docs)
        db = create_vectorstore(chunks, embedding)

    retriever = get_retriever(db)
    llm = load_llm()

    return retriever, llm

retriever, llm = setup()

query = st.text_input("Ask your question:")

if query:
    docs = retriever.invoke(query)

    st.subheader("Answer")

    def stream_response():
        for token in generate_answer(llm, query, docs):
            yield token

    st.write_stream(stream_response)

    st.subheader("Sources")
    for i, doc in enumerate(docs):
        st.markdown(f"**Source {i+1}:**")
        # st.caption(doc.page_content[:300] + "...")
        st.caption(doc.page_content)
    st.markdown("---Done---")