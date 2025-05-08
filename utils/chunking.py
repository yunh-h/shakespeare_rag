def chunk_text_by_words(text, max_words=400, overlap=40):
    """
    Chunk text into fixed word sizes with overlap.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words - overlap):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))

    return chunks