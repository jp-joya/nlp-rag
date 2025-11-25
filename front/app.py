from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_gemini import rag_answer
import re

app = Flask(__name__)
CORS(app)

def parse_references(context_text):
    """Extrae referencias del contexto de texto con su contenido."""
    references = []
    
    # Busca patrones como:
    # - [Fuente: nombre_archivo]
    # contenido del texto
    # (hasta la siguiente [Fuente: o fin del texto)
    pattern = r'\[Fuente:\s*([^\]]+)\]\s*\n(.*?)(?=\[Fuente:|$)'
    matches = re.findall(pattern, context_text, re.DOTALL)
    
    seen_sources = set()
    for source, content in matches:
        source = source.strip()
        if source not in seen_sources:  # evitar duplicados
            seen_sources.add(source)
            references.append({
                'source': source,
                'content': content.strip(),
                'type': 'texto'
            })
    
    return references

def parse_images(context_images):
    """Extrae rutas de imágenes del contexto de imágenes."""
    # Busca patrones como [Imagen: ruta/imagen.jpg]
    pattern = r'\[Imagen:\s*([^\]]+)\]'
    image_matches = re.findall(pattern, context_images)
    
    # Extraer solo el nombre del archivo (último componente de la ruta)
    import os
    image_names = [os.path.basename(img.strip()) for img in image_matches]
    
    # También busca descripciones
    description_pattern = r'Descripción:\s*([^\n]+)'
    descriptions = re.findall(description_pattern, context_images)
    
    return image_names, descriptions

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'Query vacía'}), 400
        
        result = rag_answer(user_query)
        
        # Extraer referencias e imágenes
        references = parse_references(result['context_text'])
        image_paths, image_descriptions = parse_images(result['context_images'])
        
        # Construir lista de imágenes con sus descripciones
        images_with_desc = []
        for idx, path in enumerate(image_paths):
            description = image_descriptions[idx] if idx < len(image_descriptions) else 'Sin descripción'
            images_with_desc.append({
                'path': path.strip(),
                'description': description.strip()
            })
        
        return jsonify({
            'query': result['query'],
            'answer': result['answer'],
            'context_text': result['context_text'][:500],
            'context_images': result['context_images'][:500],
            'references': references,
            'images': images_with_desc
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<path:filename>', methods=['GET'])
def get_image(filename):
    """Sirve imágenes desde el directorio data/raw/images/"""
    import os
    from flask import send_file
    
    image_path = os.path.join('./data/raw/images', filename)
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Imagen no encontrada'}), 404
    
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True, port=5000)
