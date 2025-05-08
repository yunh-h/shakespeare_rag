import json
import faiss
from sentence_transformers import SentenceTransformer
from utils.retrieve import *
from utils.openai import generate_answer_with_gpt

# Load resources
def load_resources():
    index = faiss.read_index("data/faiss_index_400w40o.bin")

    with open("data/all_chunks_400w40o_with_metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return index, metadata, embedder

# Process a query
def process_query(query, index, metadata, embedder, top_k=5):
    results = retrieve_passages_with_keywords(query, top_k)
    answer = generate_answer_with_gpt(query, results)
    return results, answer

if __name__ == "__main__":
    # Load resources
    index, metadata, embedder = load_resources()

    # Example query
    query = "Which sonnet has the line 'shall I compare thee to a summer's day'?"
    results, answer = process_query(query, index, metadata, embedder)

    # Print results
    print("Generated Answer:")
    print(answer)

    print("Top Retrieved Passages:")
    for result in results:
        print(f"Title: {result['metadata']['title']}")
        print(f"Contents: {result['contents']}")
        print("---")

    