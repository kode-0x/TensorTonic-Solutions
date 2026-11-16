def word_count_dict(sentences: list) -> dict:
    """
    Returns a dictionary of token counts.
    """
    word_count = {}
    for sentence in sentences:
        for word in sentence:
            word_count[word] = word_count.get(word, 0) + 1
    return word_count