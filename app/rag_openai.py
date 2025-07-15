import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai.embeddings import OpenAIEmbeddings
from app.prompt_templates import custom_prompt

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_rag_chain_openai(vectorstore_path="vectorstore/"):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-nano")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return chain
