def text_chunking(tokens: list, chunk_size: int, overlap: int) -> list:
    """
    Returns fixed-size token chunks with the requested overlap.
    """
    step = chunk_size - overlap
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens) - overlap, step)]