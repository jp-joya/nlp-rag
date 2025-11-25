import os
import glob
import uuid
import chromadb
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# ======================
# CONFIG
# ======================
IMG_PATH = "data/raw/images"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "nutricion_imagenes"

BLIP_MODEL = "Salesforce/blip-image-captioning-large"


# ======================
# CAPTION GENERATOR
# ======================

def generate_caption(processor, model, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[ERROR] No se pudo procesar {image_path}: {e}")
        return None


# ======================
# CHECK EXISTING IMAGES
# ======================

def get_already_ingested_images(collection):
    """
    Devuelve un conjunto con los nombres de archivos ya registrados.
    """
    results = collection.get(include=["metadatas"], limit=999999)
    ingested = set()

    for meta in results.get("metadatas", []):
        img = meta.get("source_image")
        if img:
            ingested.add(os.path.basename(img))  # solo nombre, no ruta

    return ingested


# ======================
# INGESTION
# ======================

def main():
    print("\n========== Ingest Imágenes ==========")

    # Cargar BLIP
    print("[INFO] Cargando modelo BLIP...")
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)

    # Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Obtener imágenes ya ingresadas
    ingested = get_already_ingested_images(collection)
    print(f"[INFO] Imágenes previamente ingresadas: {len(ingested)}")

    # Buscar nuevas imágenes
    image_files = glob.glob(os.path.join(IMG_PATH, "*"))
    image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    # Solo las nuevas
    new_images = [f for f in image_files if os.path.basename(f) not in ingested]

    print(f"[INFO] Nuevas imágenes detectadas: {len(new_images)}")

    if not new_images:
        print("[INFO] No hay imágenes nuevas para procesar.")
        return

    # Procesar imágenes nuevas
    for img in new_images:
        print(f"[INFO] Procesando {img}...")
        caption = generate_caption(processor, model, img)

        if not caption:
            continue

        img_id = str(uuid.uuid4())

        collection.add(
            ids=[img_id],
            documents=[caption],
            metadatas=[{
                "source_image": os.path.basename(img),
                "type": "image_caption"
            }]
        )

        print(f"[OK] Insertada: {os.path.basename(img)} → '{caption}'")

    print("\n[OK] Ingest de imágenes completado.\n")


if __name__ == "__main__":
    main()
