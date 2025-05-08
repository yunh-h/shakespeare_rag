import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
import re
import os

embedder = SentenceTransformer('all-MiniLM-L6-v2')
_chunks_with_metadata = None

def load_chunks_with_metadata(file_path="data/all_chunks_400w40o_with_metadata.json"):
    """
    Load chunks with metadata from a JSON file. Use a global variable to avoid reloading.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of chunks with metadata.
    """
    global _chunks_with_metadata

    if _chunks_with_metadata is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            _chunks_with_metadata = json.load(file)
    return _chunks_with_metadata

def retrieve_passages(query, top_k=5, filter_title=None, filter_genre=None):
    """
    Retrieve passages based on the query, with optional metadata filtering by title and genre.

    Args:
        query (str): The user query.
        top_k (int): Number of top results to return.
        filter_title (str): Optional title to filter chunks by (e.g., "Hamlet").
        filter_genre (str): Optional genre to filter chunks by (e.g., "Sonnet", "Poem", "Play").

    Returns:
        list: Retrieved passages.
    """
    chunks_with_metadata = load_chunks_with_metadata()
    filtered_chunks = chunks_with_metadata
    
    if filter_title:
        filtered_chunks = [
            chunk for chunk in filtered_chunks
            if filter_title.lower() in chunk["metadata"]["title"].lower()
        ]
    if filter_genre:
        filtered_chunks = [
            chunk for chunk in filtered_chunks
            if chunk["metadata"]["genre"].lower() == filter_genre.lower()
        ]

    if not filtered_chunks:
        filtered_chunks = chunks_with_metadata
        
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    filtered_embeddings = np.array([
        embedder.encode(chunk["contents"], convert_to_numpy=True)
        for chunk in filtered_chunks
    ])

    # Create a temporary FAISS index for the filtered chunks
    dimension = filtered_embeddings.shape[1]
    temp_index = faiss.IndexFlatIP(dimension)
    temp_index.add(filtered_embeddings)

    distances, indices = temp_index.search(query_embedding, top_k)

    # Retrieve the top results
    results = []
    for i, idx in enumerate(indices[0]):
        result = filtered_chunks[idx]
        results.append({
            "rank": i + 1,
            "id": result["id"],
            "name": result["name"],
            "contents": result["contents"],
            "distance": distances[0][i]
        })

    return results

# Predefined lists of names and genres
names_list = ["all's well that ends well", "antony and cleopatra", "as you like it", "comedy of errors", "coriolanus", "cymbeline", "hamlet", "henry iv", "henry v", "henry vi", "henry viii", "king john", "julius caesar", "king lear", "love's labour's lost", "macbeth", "measure for measure", "the merchant of venice", "the merry wives of windsor", "a midsummer night's dream", "much ado about nothing", "othello", "pericles, prince of tyre", "richard ii", "richard iii", "romeo and juliet", "the taming of the shrew", "titus andronicus", "trolius and cressida", "twelfth night", "the two gentlemen of verona", "the two noble kinsmen", "the winter's tale", "a lover's complaint", "the passionate pilgrim", "the phoenix and the turtle", "the rape of lucrece", "venus and adonis", "sonnet"]
genres_list = ["Sonnet", "Poem", "Play"]

def extract_keywords(query):
    """
    Extract keywords for names and genres from the query.

    Args:
        query (str): The user query.

    Returns:
        dict: A dictionary with detected name and genre keywords.
    """
    query_lower = query.lower()
    detected_names = []
    detected_genres = []

    for name in names_list:
        if name.lower() == "sonnet":
            # Check if "sonnet" is followed by a number
            if re.search(r"sonnet\s*\d+", query_lower):
                detected_names.append(re.search(r"sonnet\s+\d+", query_lower).group())
        elif name.lower() in query_lower:
            detected_names.append(name)

    for genre in genres_list:
        if genre.lower() == "sonnet":
            if "sonnet" in query_lower and not re.search(r"sonnet\s+\d+", query_lower):
                detected_genres.append("Sonnet")
        elif genre.lower() in query_lower:
            detected_genres.append(genre)

    return {
        "names": detected_names,
        "genres": detected_genres
    }

def retrieve_passages_with_keywords(query, top_k=5):
    """
    Retrieve passages based on the query, automatically detecting names and genres.

    Args:
        query (str): The user query.
        top_k (int): Number of top results to return.

    Returns:
        list: Retrieved passages.
    """
    keywords = extract_keywords(query)
    filter_title = keywords["names"][0] if keywords["names"] else None
    filter_genre = keywords["genres"][0] if keywords["genres"] else None

    return retrieve_passages(query, top_k=top_k, filter_title=filter_title, filter_genre=filter_genre)
