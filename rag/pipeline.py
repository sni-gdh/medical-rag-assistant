from llama_cpp import Llama
from config import MODEL_PATH
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio
from rag.retriever import get_retriever
from rag.vector_store import load_vectorstore
from rag.embed import get_embedding


def load_llm():
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=6,
        n_batch = 512 
    )
    return llm

app = FastAPI()

embedding = get_embedding()
db = load_vectorstore(embedding)
retriever = get_retriever(db)
llm = load_llm()


def generate_answer(llm, query, docs):
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
                You are a biomedical assistant. Answer ONLY from the context.

                Context:
                {context}

                Question:
                {query}

                Answer:
            """

    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.7,
        top_p=0.9,
        stream=True   # 🔥 IMPORTANT
    )
    print(output)
    for chunk in output:
        yield chunk["choices"][0]["text"]


@app.get("/stream")
async def stream(query: str):

    docs = retriever.invoke(query)

    async def event_generator():
        for token in generate_answer(llm, query, docs):
            yield {
                "event": "message",
                "data": token
            }
            await asyncio.sleep(0.01)

    return EventSourceResponse(event_generator())