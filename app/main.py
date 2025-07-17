from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_llama import get_rag_chain_llama
#from app.rag_openai import get_rag_chain_openai


app = FastAPI(
    title="RAG Knowledge Assistant",
    description="Consulta documentos usando OpenAI y recuperación semántica",
    version="1.0"
)

# Modelo de entrada
class Question(BaseModel):
    question: str

# Inicializar la cadena RAG
qa_chain = get_rag_chain_llama() 

@app.post("/ask")
def ask_question(query: Question):
    respuesta = qa_chain.invoke(query.question)
    return {"respuesta": respuesta}
