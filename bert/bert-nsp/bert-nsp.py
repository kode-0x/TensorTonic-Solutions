def create_nsp_pairs(documents: list, pair_specs: list) -> list:
    """
    Returns sentence_a, sentence_b, and is_next dictionaries in a list.
    """
    result = []

    for spec in pair_specs:
        doc_a, sent_a = spec["doc_a"], spec["sent_a"]
        doc_b, sent_b = spec["doc_b"], spec["sent_b"]

        result.append({
            "sentence_a": documents[doc_a][sent_a],
            "sentence_b": documents[doc_b][sent_b],
            "is_next": int(doc_a == doc_b and sent_b == sent_a + 1)
        })

    return result