from app.rag_llama import get_rag_chain_llama
chain = get_rag_chain_llama()
#respuesta = chain.invoke("¿Qué dice el documento, resumir en 100 palabras?")
#print(respuesta)

# ...existing code...
texto = input("Introduce el texto a procesar: ")
respuesta = chain.invoke(texto)
print(respuesta["result"] if isinstance(respuesta, dict) and "result" in respuesta else respuesta)
# ...existing code...



