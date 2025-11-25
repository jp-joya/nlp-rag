import os
import glob
import chromadb
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

CHROMA_PATH = "./chroma_db"
IMAGE_PATH = "data/raw/images"

def load_blip_model():
    print("[INFO] Cargando modelo BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    print("[INFO] Modelo BLIP cargado correctamente.")
    return processor, model

def generate_caption(processor, model, img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs, max_length=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[ERROR] No se pudo procesar {img_path}: {e}")
        return None

def store_captions_in_chroma(captions):
    print("[INFO] Conectando a Chroma...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name="nutricion_imagenes",
        metadata={"description": "Captions generadas con BLIP para imágenes de nutrición"}
    )

    print("[INFO] Insertando captions en Chroma...")

    ids = []
    docs = []
    metas = []

    for idx, (img, caption) in enumerate(captions.items()):
        ids.append(f"img_{idx}")
        docs.append(caption)
        metas.append({
            "source_image": img,
            "type": "image_caption"
        })

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas
    )

    print("[INFO] Captions insertadas exitosamente.")

def main():
    # 1. Load model
    processor, model = load_blip_model()

    # 2. Process images
    images = glob.glob(os.path.join(IMAGE_PATH, "*.jpg"))
    print(f"[INFO] Imágenes encontradas: {len(images)}")

    captions = {}

    for img in images:
        print(f"[INFO] Procesando imagen: {img}")
        caption = generate_caption(processor, model, img)
        if caption:
            print(f"   → Caption: {caption}")
            captions[img] = caption

    # 3. Store in Chroma
    store_captions_in_chroma(captions)

if __name__ == "__main__":
    main()
