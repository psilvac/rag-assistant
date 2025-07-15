from app.rag_llama import get_rag_chain_llama
chain = get_rag_chain_llama()
#respuesta = chain.invoke("¿Qué dice el documento, resumir en 100 palabras?")
#print(respuesta)

# ...existing code...
respuesta = chain.invoke("¿cuales son las instrucciones de pago?")
print(respuesta["result"] if isinstance(respuesta, dict) and "result" in respuesta else respuesta)
# ...existing code...