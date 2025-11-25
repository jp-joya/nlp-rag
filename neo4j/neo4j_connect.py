from neo4j import GraphDatabase

URI = "neo4j://34.230.13.44:7687"
USER = "neo4j"
PASSWORD = "adminadmin"

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j está funcionando' as msg")
        print(result.single()["msg"])

if __name__ == "__main__":
    test_connection()
