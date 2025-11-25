import os
import glob
import chromadb
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# ------------ CONFIGURACIÓN ------------------

DATA_PATH = "data/raw/text"
CHROMA_PATH = "./chroma_db"

# Embeddings recomendados (rápidos y buenos)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------


def load_markdown_files():
    files = glob.glob(os.path.join(DATA_PATH, "*.md"))
    documents = []

    for file in files:
        loader = UnstructuredMarkdownLoader(file)
        docs = loader.load()

        # Agregamos metadatos útiles
        for d in docs:
            d.metadata["source"] = os.path.basename(file)

        documents.extend(docs)

    print(f"[INFO] Archivos cargados: {len(documents)}")
    return documents


def recursive_chunking(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Chunks creados (Recursive): {len(chunks)}")
    return chunks


def semantic_chunking(docs):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chunker = SemanticChunker(embeddings)
    chunks = chunker.split_documents(docs)
    print(f"[INFO] Chunks creados (Semantic): {len(chunks)}")
    return chunks


def store_in_chroma(docs):
    print("[INFO] Conectando a ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name="nutricion_textos",
        metadata={"description": "Chunks de documentos de nutrición"}
    )

    ids = [f"doc_{i}" for i in range(len(docs))]
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    print("[INFO] Insertando en Chroma...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    print("[INFO] Datos insertados correctamente.")


def main():
    # 1. Cargar documentos originales
    docs = load_markdown_files()

    # 2. Chunking híbrido: recursion + semantic
    chunks_recursive = recursive_chunking(docs)
    chunks_semantic = semantic_chunking(docs)

    # 3. Unir ambos tipos de chunks
    final_chunks = chunks_recursive + chunks_semantic

    print(f"[INFO] Total de chunks preparados: {len(final_chunks)}")

    # 4. Guardar en Chroma
    store_in_chroma(final_chunks)


if __name__ == "__main__":
    main()
