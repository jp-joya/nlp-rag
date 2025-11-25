import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="test_collection"
)

collection.add(
    ids=["1"],
    documents=["Este es un primer documento de prueba sobre nutrición."],
    metadatas=[{"tema": "nutricion", "tipo": "texto"}]
)

result = collection.query(
    query_texts=["¿Qué dice el documento?"],
    n_results=1
)

print(result)
