import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("nutricion_imagenes")

query = collection.query(
    query_texts=["imagen de un vaso de agua"],
    n_results=3
)

print(query)
