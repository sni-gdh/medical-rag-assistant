from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBED_MODEL

def get_embedding():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)