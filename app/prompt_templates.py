from langchain.prompts import PromptTemplate

# Prompt personalizado para el sistema RAG
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Eres un asistente experto en interpretación de documentos. 
Lee cuidadosamente el siguiente contexto y responde con claridad, precisión y lenguaje formal.

Contexto relevante:
{context}

Pregunta del usuario:
{question}

Respuesta:
""")
