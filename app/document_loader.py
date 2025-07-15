import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def cargar_y_vectorizar_documentos(ruta_docs="data/documentos/", ruta_salida="vectorstore/"):

    documentos = []

    for archivo in os.listdir(ruta_docs):
        if archivo.endswith(".pdf"):
            print(f"Cargando archivo: {archivo}")
            loader = PyPDFLoader(os.path.join(ruta_docs, archivo))
            documentos.extend(loader.load())

    # Dividir los documentos en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documentos)

    # Crear embeddings con OpenAI
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Crear vectorstore y guardar localmente
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(ruta_salida)

    print(f"Vectorstore guardado en: {ruta_salida}")
