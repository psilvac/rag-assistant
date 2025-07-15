from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from app.prompt_templates import custom_prompt

def get_rag_chain_llama(vectorstore_path="vectorstore/"):
    # Cargar FAISS con embeddings open-source
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    # Conectar con modelo LLaMA local en Ollama
    llm = OllamaLLM(model="llama3", temperature=0.1)

    # Construir cadena RAG
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return chain
