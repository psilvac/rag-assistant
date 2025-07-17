# 🧠 RAG Knowledge Assistant

Smart assistant to query PDF documents using natural language and LLMs. Implements RAG (Retrieval-Augmented Generation) with Prompt Engineering, comparing closed (GPT-4) and open-source (LLaMA/Mistral) models.

## 📌 Project Description

This project demonstrates how to build a GenAI system using Python and LangChain to answer questions from documents, integrating vector search (FAISS), prompt design, and LLMs (OpenAI or HuggingFace).

## 🛠️ Technologies
- Python 3.10+
  
- LangChain
  
- OpenAI (GPT-4)
  
- FAISS
  
- FastAPI
  
- HuggingFace Transformers
  
- PyPDF
  
- dotenv
  
- Docker (optional)

## 🚀 How to Run

```bash
# 1. Clone repository
git clone https://github.com/youruser/rag-knowledge-assistant.git
cd rag-knowledge-assistant

# 2. Create virtual env and install dependencies
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# 3. Put your PDFs in /data/documentos/
# 4. Build vector store
python -i
>>> from app.document_loader import cargar_y_vectorizar_documentos
>>> cargar_y_vectorizar_documentos()

# 5. Run the API
uvicorn app.main:app --reload
```

## 📬 API Usage

Access docs at:
➡️ http://localhost:8000/docs

Send a POST like:
```bash
{
  "question": "What does the document say about the contract duration?"
}
```
## 🔍 Model Comparison
Easily switch between:

- 🔒 GPT-4 (OpenAI)

- 🔓 LLaMA / Mistral (HuggingFace)

## ✨ Suggested Extensions
- File upload via API

- Streamlit frontend

- Chat history logging

- Cloud deployment (Azure, AWS, HuggingFace Spaces)
