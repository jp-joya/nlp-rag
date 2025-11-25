import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    TextLoader
)


# =============================
# CONFIG
# =============================

DATA_PATH = "data/raw/text"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "nutricion_textos"

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


# =============================
# EMBEDDING WRAPPER PARA SemanticChunker
# =============================

class STEmbeddingWrapper:
    """
    Wrapper para SentenceTransformer compatible con SemanticChunker.
    """
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()


# =============================
# CARGA DE DOCUMENTOS
# =============================

def load_all_documents():
    """
    Carga .md y .txt desde la carpeta DATA_PATH.
    Retorna una lista de langchain Documents, con metadata del archivo.
    """
    docs = []
    
    md_files = glob.glob(os.path.join(DATA_PATH, "*.md"))
    txt_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
    
    print(f"[INFO] Archivos .md encontrados: {len(md_files)}")
    print(f"[INFO] Archivos .txt encontrados: {len(txt_files)}")

    # Markdown
    for file in md_files:
        loader = UnstructuredMarkdownLoader(file)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = os.path.basename(file)
        docs.extend(loaded)

    # Text files
    for file in txt_files:
        loader = TextLoader(file, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source_file"] = os.path.basename(file)
        docs.extend(loaded)

    print(f"[INFO] Documentos cargados: {len(docs)}")
    return docs


# =============================
# CHUNKING RECURSIVE
# =============================

def recursive_chunking(docs, chunk_size=400, chunk_overlap=80):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


# =============================
# CHUNKING SEMANTIC
# =============================

def semantic_chunking(docs):
    embedder = STEmbeddingWrapper(EMBEDDING_MODEL)
    chunker = SemanticChunker(embeddings=embedder)
    return chunker.split_documents(docs)


# =============================
# OBTENER ARCHIVOS YA INGRESADOS
# =============================

def get_already_ingested_files(collection):
    results = collection.get(include=["metadatas"], limit=999999)
    ingested = set()

    for meta in results.get("metadatas", []):
        file = meta.get("source_file")
        if file:
            ingested.add(file)

    return ingested


# =============================
# INGEST EN CHROMA
# =============================

def ingest_chunks(collection, chunks, new_files):
    """
    Inserta chunks solo para archivos NUEVOS.
    """
    to_insert_ids = []
    to_insert_docs = []
    to_insert_meta = []

    for i, chunk in enumerate(chunks):
        file_src = chunk.metadata.get("source_file")

        # Solo insertar si el archivo es nuevo
        if file_src not in new_files:
            continue

        to_insert_ids.append(f"{file_src}_{i}")
        to_insert_docs.append(chunk.page_content)
        to_insert_meta.append({
            "source_file": file_src,
            "type": "text_chunk"
        })

    if to_insert_ids:
        collection.add(
            ids=to_insert_ids,
            documents=to_insert_docs,
            metadatas=to_insert_meta
        )
        print(f"[INFO] Insertados {len(to_insert_ids)} chunks nuevos.")
    else:
        print("[INFO] No hay nuevos archivos para insertar.")


# =============================
# MAIN
# =============================

def main():
    print("\n========== Ingest Textos ==========")

    docs = load_all_documents()

    # Obtener lista de archivos de los documentos cargados
    all_files = sorted(list(set([d.metadata["source_file"] for d in docs])))

    # Conectar con Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    already = get_already_ingested_files(collection)

    # Archivos nuevos
    new_files = [f for f in all_files if f not in already]

    print(f"[INFO] Archivos ya ingresados: {len(already)}")
    print(f"[INFO] Archivos nuevos: {new_files}")

    if not new_files:
        print("[INFO] No hay nuevos archivos por procesar.")
        return

    # Filtrar docs para solo archivos nuevos
    docs_to_process = [d for d in docs if d.metadata["source_file"] in new_files]

    # Chunking
    print("[INFO] Generando chunks recursive...")
    chunks_r = recursive_chunking(docs_to_process)

    print("[INFO] Generando chunks semantic... puede tardar")
    chunks_s = semantic_chunking(docs_to_process)

    all_chunks = chunks_r + chunks_s
    print(f"[INFO] Total chunks generados: {len(all_chunks)}")

    # Ingest
    ingest_chunks(collection, all_chunks, new_files)

    print("\n[OK] Ingest completo.\n")


if __name__ == "__main__":
    main()
