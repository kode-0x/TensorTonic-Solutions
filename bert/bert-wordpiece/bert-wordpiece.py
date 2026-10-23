def wordpiece_tokenize(text: str, vocab: dict, unk_token: str = "[UNK]", max_word_length: int = 100) -> list:
    """
    Returns the WordPiece tokens as a list of strings.
    """
    result = []

    for word in text.lower().split():
        if len(word) > max_word_length:
            result.append(unk_token)
            continue

        pieces = []
        start = 0
        while start < len(word):
            end = len(word)
            piece = None

            while end > start:
                candidate = word[start:end]
                if start > 0:
                    candidate = "##" + candidate

                if candidate in vocab:
                    piece = candidate
                    break

                end -= 1

            if piece is None:
                pieces = [unk_token]
                break

            pieces.append(piece)
            start = end

        result.extend(pieces)

    return result