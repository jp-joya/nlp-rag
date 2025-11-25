import chromadb

CHROMA_PATH = "./chroma_db"

def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collections = client.list_collections()

    if not collections:
        print("[INFO] No hay colecciones para borrar.")
        return

    print(f"[INFO] Colecciones encontradas ({len(collections)}):")
    for c in collections:
        print(f" - {c.name}")

    print("\n[INFO] Borrando colecciones...")

    for c in collections:
        client.delete_collection(c.name)
        print(f"[OK] Borrada: {c.name}")

    print("\n[OK] TODAS las colecciones fueron eliminadas.")

if __name__ == "__main__":
    main()
