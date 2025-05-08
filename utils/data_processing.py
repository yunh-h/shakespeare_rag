import re
import json

def parse_contents(text):
    """
    Parse the 'Contents' section of the text to extract book titles and assign IDs.
    """
    # Find the "Contents" section
    contents_start = re.search(r"Contents", text, re.IGNORECASE)
    if not contents_start:
        raise ValueError("Contents section not found in the text.")
    
    # Extract the portion of the text starting from "Contents"
    contents_text = text[contents_start.end():]
    
    # Stop parsing at the next blank line or non-content section
    contents_end = re.search(r"\n\s*\n\s*\n", contents_text)
    if contents_end:
        contents_text = contents_text[:contents_end.start()]
    # print(f"Contents text: {contents_text}")  # Debugging line

    books = text[contents_start.end() + contents_end.end():].strip()
    # print(books[:1000])  # Debugging line

    # Split the contents into lines and clean up whitespace
    lines = [line.strip() for line in contents_text.split("\n") if line.strip()]
    # print(f"Lines: {lines}")  # Debugging line
    
    # Initialize the result dictionary
    parsed_contents = []
    parsed_dict = {}
    
    for idx, line in enumerate(lines):
        # For each title, create a unique ID by normalizing the title
        book_id = re.sub(r"[^a-z0-9_]+", "_", line.lower().strip())
        parsed_contents.append({"id": f"book_{idx + 1}", "name": line})
        parsed_dict[idx+1] = line
    
    return parsed_contents, parsed_dict, books

def extract_books_from_index(text, index_dict):
    """
    Extract books and their contents using the provided index dictionary.
    """
    books = []
    sorted_keys = list(index_dict.keys())

    for i, key in enumerate(sorted_keys):
        book_name = index_dict[key]
        book_id = f"book_{key-1}" if book_name.lower() != "the sonnets" else "sonnets"

        # Find the start of the book
        book_start = text.find(book_name)
        if book_start == -1:
            raise ValueError(f"Book title '{book_name}' not found in the text.")

        # Find the end of the book (start of the next book or end of text)
        if i + 1 < len(sorted_keys):
            next_book_name = index_dict[sorted_keys[i + 1]]
            book_end = text.find(next_book_name)
        else:
            book_end = len(text)

        # Extract the book content
        book_content = text[book_start:book_end].strip()
        # print(book_content)  # Debugging line

        # Special handling for "THE SONNETS"
        if book_name.lower() == "the sonnets":
            books.extend(parse_sonnets(book_content))
        else:
            books.append({
                "id": book_id,
                "name": book_name,
                "contents": book_content
            })

    return books


def parse_sonnets(sonnets_text):
    """
    Parse the sonnets section and return a list of individual sonnet entries.
    """
    # Split the text into sections based on the sonnet number pattern
    sonnet_matches = re.split(r"\n\s*(\d+)\n\n", sonnets_text)

    # Parse each sonnet into a dictionary
    sonnets = []
    for i in range(1, len(sonnet_matches), 2):
        sonnet_number = sonnet_matches[i]
        sonnet_content = sonnet_matches[i + 1].strip()
        sonnets.append({
            "id": f"sonnet_{sonnet_number}",
            "name": f"Sonnet {sonnet_number}",
            "contents": sonnet_content
        })

    # Return the list of individual sonnet entries
    return sonnets


def save_to_json(data, output_path):
    """
    Save the parsed contents to a JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
