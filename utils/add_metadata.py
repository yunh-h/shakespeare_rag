import json

def add_metadata_to_chunks(input_path, output_path):
    """
    Add metadata (title, author, genre) to each chunk in the input JSON file and save the updated chunks to a new file.

    Args:
        input_path (str): Path to the input JSON file containing chunks.
        output_path (str): Path to save the updated JSON file with metadata.
    """
    with open(input_path, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    updated_chunks = []

    for chunk in chunks:
        # Determine genre based on the chunk ID
        if "sonnet" in chunk["id"]:
            genre = "Sonnet"
        elif "poem" in chunk["id"]:
            genre = "Poem"
        else:
            genre = "Play"

        # Extract title
        title = chunk["name"].split(" (")[0] if genre in ["Play", "Poem"] else chunk["name"]

        # Add metadata
        metadata = {
            "metadata": {
                "title": title,
                "author": "William Shakespeare",
                "genre": genre
            }
        }
        chunk.update(metadata)
        updated_chunks.append(chunk)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(updated_chunks, file, ensure_ascii=False, indent=4)

    print(f"Updated chunks saved to {output_path}")

if __name__ == "__main__":
    input_path = "all_chunks_400w40o.json"
    output_path = "all_chunks_400w40o_with_metadata.json"
    add_metadata_to_chunks(input_path, output_path)