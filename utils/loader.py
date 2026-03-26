import os
from langchain_community.document_loaders import TextLoader

def load_documents(path):
    docs = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())

    return docs