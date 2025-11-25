import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

col = client.get_collection("nutricion_imagenes")

results = col.get()

print("Total imágenes insertadas:", len(results["ids"]))
print("\nListado de imágenes:")

for i, (id_, meta) in enumerate(zip(results["ids"], results["metadatas"])):
    print(f"{i} | ID: {id_} | meta: {meta}")
