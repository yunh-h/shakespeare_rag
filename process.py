from utils.chunking import *
from utils.data_processing import *
from utils.retrieve import *
from utils.openai import *
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss

### 
# Parse the contents of a text file, extract book titles and their contents, and save the results to JSON files.
###
with open("pg100.txt", "r", encoding="utf-8") as file:
    text = file.read()
    contents, index_dict, book_content = parse_contents(text)
    save_to_json(contents, "parsed_contents.json")
    print(f"Parsed contents saved to parsed_contents.json")

    # Extract books and their contents
    books = extract_books_from_index(book_content, index_dict)
    # Save the parsed contents to a JSON file
    save_to_json(book_content, "books.json")
    print(f"Books saved to books.json")
    save_to_json(books, "books_with_sonnets.json")
    print(f"Books with sonnets saved to books_with_sonnets.json.")

# Chunk the text into smaller pieces for processing
with open("books_with_sonnets.json", "r", encoding="utf-8") as file:
    books = json.load(file)

chunks = []

for book in books:
    if "sonnet" in book["id"]:
        # Treat each sonnet as a single chunk
        chunks.append({
            "id": book["id"],
            "name": book["name"],
            "contents": book["contents"]
        })
    else:
        # Chunk longer books (plays) by token size
        # book_chunks = chunk_text(book["contents"], max_tokens=300, overlap=50)
        book_chunks = chunk_text_by_words(book["contents"])

        for idx, chunk in enumerate(book_chunks):
            chunks.append({            
                "id": f"{book['id']}_chunk_{idx + 1}",
                "name": f"{book['name']} (Chunk {idx + 1})",
                "contents": chunk
            })

# Save the combined chunks to a new JSON file
with open("all_chunks_300w.json", "w", encoding="utf-8") as output_file:
    json.dump(chunks, output_file, ensure_ascii=False, indent=4)

print(f"Total chunks created: {len(chunks)}")

# Create embeddings for the chunks using the OpenAI API
# Load a pre-trained embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load the chunks
with open("all_chunks_300w.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Generate embeddings for each chunk
embeddings = []

for chunk in chunks:
    embedding = embedder.encode(chunk["contents"], convert_to_numpy=True)
    embeddings.append(embedding)

# Convert to a NumPy array
embeddings = np.array(embeddings)
print(f"Generated embeddings for {len(embeddings)} chunks.")

# FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) # Inner product index
index.add(embeddings)

print(f"FAISS index contains {index.ntotal} embeddings.")

# Save the FAISS index to a file
faiss_outpath = "faiss_index.bin"
faiss.write_index(index, faiss_outpath)
print(f"FAISS index saved to {faiss_outpath}.")

# Adding metadata to the index
with open("all_chunks_300w.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

updated_chunks = []

for chunk in chunks:
    genre = "Play" if "sonnet" not in chunk["id"] else "Sonnet"
    title = chunk["name"].split(" (")[0] if genre == "Play" else chunk["name"]

    metadata = {
        "metadata": {
            "title": title,
            "author": "William Shakespeare",
            "genre": genre
        }
    }
    chunk.update(metadata)
    updated_chunks.append(chunk)

with open("all_chunks_300w_with_metadata.json", "w", encoding="utf-8") as file:
    json.dump(updated_chunks, file, ensure_ascii=False, indent=4)

print(f"Updated chunks saved to all_chunks_300w_with_metadata.json")


### from copilot
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils.chunking import chunk_text_by_words

def process_text_file(input_file, output_chunks_file, output_metadata_file, faiss_index_file):
    # Step 1: Read the text file
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()

    # Step 2: Parse the contents (assuming `parse_contents` is defined elsewhere)
    contents, index_dict, book_content = parse_contents(text)

    # Step 3: Extract books and their contents
    books = extract_books_from_index(book_content, index_dict)

    # Step 4: Chunk the text
    chunks = []
    for book in books:
        if "sonnet" in book["id"]:
            # Treat each sonnet as a single chunk
            chunks.append({
                "id": book["id"],
                "name": book["name"],
                "contents": book["contents"]
            })
        else:
            # Chunk longer books (plays) by word size
            book_chunks = chunk_text_by_words(book["contents"])
            for idx, chunk in enumerate(book_chunks):
                chunks.append({
                    "id": f"{book['id']}_chunk_{idx + 1}",
                    "name": f"{book['name']} (Chunk {idx + 1})",
                    "contents": chunk
                })

    # Step 5: Save chunks to a JSON file
    with open(output_chunks_file, "w", encoding="utf-8") as output_file:
        json.dump(chunks, output_file, ensure_ascii=False, indent=4)

    print(f"Chunks saved to {output_chunks_file}")

    # Step 6: Generate embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = [embedder.encode(chunk["contents"], convert_to_numpy=True) for chunk in chunks]
    embeddings = np.array(embeddings)

    # Step 7: Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, faiss_index_file)
    print(f"FAISS index saved to {faiss_index_file}")

    # Step 8: Add metadata to chunks
    updated_chunks = []
    for chunk in chunks:
        genre = "Play" if "sonnet" not in chunk["id"] else "Sonnet"
        title = chunk["name"].split(" (")[0] if genre == "Play" else chunk["name"]
        metadata = {
            "metadata": {
                "title": title,
                "author": "William Shakespeare",
                "genre": genre
            }
        }
        chunk.update(metadata)
        updated_chunks.append(chunk)

    # Step 9: Save updated chunks with metadata
    with open(output_metadata_file, "w", encoding="utf-8") as file:
        json.dump(updated_chunks, file, ensure_ascii=False, indent=4)

    print(f"Metadata saved to {output_metadata_file}")

# Run the data processing pipeline
if __name__ == "__main__":
    process_text_file(
        input_file="pg100.txt",
        output_chunks_file="all_chunks_300w.json",
        output_metadata_file="all_chunks_300w_with_metadata.json",
        faiss_index_file="faiss_index.bin"
    )
