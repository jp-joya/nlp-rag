import os
import uuid
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------
NEO4J_URI = "neo4j://34.230.13.44:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "adminadmin"
DATA_TEXT_DIR = "data/raw/text"
DATA_IMG_DIR = "data/raw/images"
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------------
# Conexión Neo4j
# ------------------------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ------------------------------------------
# Utilidades
# ------------------------------------------
def document_exists(session, name):
    result = session.run("MATCH (d:Document {name:$name}) RETURN d", name=name)
    return result.single() is not None

def image_exists(session, path):
    result = session.run("MATCH (i:Image {path:$path}) RETURN i", path=path)
    return result.single() is not None

def embed_vector(model, text):
    return model.encode(text).tolist()

# ------------------------------------------
# Procesamiento Texto
# ------------------------------------------
def ingest_text():
    print("\n### INGESTANDO DOCUMENTOS ###")

    embedder = SentenceTransformer(EMBED_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

    with driver.session() as session:
        for filename in os.listdir(DATA_TEXT_DIR):
            path = os.path.join(DATA_TEXT_DIR, filename)

            if not filename.endswith((".txt", ".md")):
                continue

            if document_exists(session, filename):
                print(f"[SKIP] Documento ya existe: {filename}")
                continue

            print(f"[OK] Ingestando documento: {filename}")

            content = open(path, "r", encoding="utf-8").read()
            chunks = splitter.split_text(content)

            # Crear nodo Document
            doc_id = str(uuid.uuid4())
            session.run("""
                CREATE (d:Document {id:$id, name:$name})
            """, id=doc_id, name=filename)

            # Insertar chunks + embeddings
            for idx, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                embed = embed_vector(embedder, chunk)

                session.run("""
                    CREATE (c:Chunk {id:$cid, text:$txt})
                    WITH c
                    MATCH (d:Document {id:$did})
                    CREATE (d)-[:HAS_CHUNK]->(c)
                """, cid=chunk_id, txt=chunk, did=doc_id)

                emb_id = str(uuid.uuid4())
                session.run("""
                    MATCH (c:Chunk {id:$cid})
                    CREATE (e:Embedding {id:$eid, vector:$vec})
                    CREATE (c)-[:HAS_EMBEDDING]->(e)
                """, cid=chunk_id, eid=emb_id, vec=embed)

    print("✓ Ingesta de documentos completada.\n")

# ------------------------------------------
# Procesamiento Imagen
# ------------------------------------------
def caption_image(image_path):
    """
    Llamada a BLIP para generar caption.
    BLIP se usa a través de transformers pipeline.
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from PIL import Image

    processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)

    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


def ingest_images():
    print("\n### INGESTANDO IMÁGENES ###")

    embedder = SentenceTransformer(EMBED_MODEL)

    with driver.session() as session:
        for filename in os.listdir(DATA_IMG_DIR):
            path = os.path.join(DATA_IMG_DIR, filename)

            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            if image_exists(session, path):
                print(f"[SKIP] Imagen ya existe: {filename}")
                continue

            print(f"[OK] Procesando imagen: {filename}")

            # Crear nodo Image
            img_id = str(uuid.uuid4())
            session.run("""
                CREATE (i:Image {id:$id, path:$path})
            """, id=img_id, path=path)

            # Caption
            caption = caption_image(path)
            cap_id = str(uuid.uuid4())

            session.run("""
                MATCH (i:Image {id:$iid})
                CREATE (c:Caption {id:$cid, text:$txt})
                CREATE (i)-[:HAS_CAPTION]->(c)
            """, iid=img_id, cid=cap_id, txt=caption)

            # Embedding del caption
            emb = embed_vector(embedder, caption)
            emb_id = str(uuid.uuid4())

            session.run("""
                MATCH (c:Caption {id:$cid})
                CREATE (e:Embedding {id:$eid, vector:$vec})
                CREATE (c)-[:HAS_EMBEDDING]->(e)
            """, cid=cap_id, eid=emb_id, vec=emb)

    print("✓ Ingesta de imágenes completada.\n")


# ------------------------------------------
# MAIN
# ------------------------------------------
if __name__ == "__main__":
    ingest_text()
    ingest_images()
