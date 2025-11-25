from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

NEO4J_URI = "neo4j://34.230.13.44:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "adminadmin"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embedder = SentenceTransformer(EMBED_MODEL)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def embed_query(query: str):
    return embedder.encode(query).tolist()


def retrieve_from_neo4j(query: str, top_k: int = 5):
    vector = embed_query(query)

    cypher = """
    WITH $query_vector AS query
    MATCH (e:Embedding)

    WITH e, query,
         reduce(dot = 0.0, i IN range(0, size(query)-1) |
             dot + query[i] * e.vector[i]
         ) AS dot,
         reduce(mq = 0.0, i IN range(0, size(query)-1) |
             mq + query[i]^2
         ) AS mag_query,
         reduce(me = 0.0, i IN range(0, size(e.vector)-1) |
             me + e.vector[i]^2
         ) AS mag_emb

    WITH e,
         dot / (sqrt(mag_query) * sqrt(mag_emb)) AS score
    ORDER BY score DESC
    LIMIT $k

    MATCH (owner)-[:HAS_EMBEDDING]->(e)
    RETURN owner, score;
    """

    with driver.session() as session:
        result = session.run(cypher, query_vector=vector, k=top_k)

        outputs = []
        for record in result:
            owner = record["owner"]
            score = record["score"]

            if "Chunk" in owner.labels:
                outputs.append({
                    "type": "text",
                    "score": score,
                    "content": owner["text"]
                })

            elif "Caption" in owner.labels:
                outputs.append({
                    "type": "image_caption",
                    "score": score,
                    "content": owner["text"]
                })

        return outputs


if __name__ == "__main__":
    q = "por qué es importante dormir bien"
    hits = retrieve_from_neo4j(q, top_k=5)

    print("\n=== TOP RESULTS ===")
    for h in hits:
        print(f"\n[{h['type'].upper()}] Score: {h['score']:.4f}")
        print(h['content'])
