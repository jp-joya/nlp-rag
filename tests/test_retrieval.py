import chromadb

CHROMA_PATH = "./chroma_db"
TEXT_COLLECTION = "nutricion_textos"
IMAGE_COLLECTION = "nutricion_imagenes"

def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    col_text = client.get_collection(TEXT_COLLECTION)
    col_img = client.get_collection(IMAGE_COLLECTION)

    query = "¿Por qué es importante dormir bien?"

    print("\n====== BUSCANDO TEXTO ======\n")
    t = col_text.query(query_texts=[query], n_results=5)
    for i, doc in enumerate(t["documents"][0]):
        print(f"[TEXT {i}] {doc}\n")
    print("Distancias:", t["distances"][0])

    print("\n====== BUSCANDO IMÁGENES ======\n")
    i = col_img.query(query_texts=[query], n_results=5)
    for idx, (doc, dist, meta) in enumerate(zip(i["documents"][0], i["distances"][0], i["metadatas"][0])):
        print(f"[IMG {idx}] {doc}")
        print("  distancia:", dist)
        print("  metadata:", meta)
        print()

if __name__ == "__main__":
    main()
