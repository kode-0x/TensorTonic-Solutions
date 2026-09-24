class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        self.word_to_id = {
            token: i for i, token in enumerate(special_tokens)
        }

        words = set()
        for text in texts:
            words.update(text.lower().split())

        for word in sorted(words):
            self.word_to_id[word] = len(self.word_to_id)

        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        words = text.lower().split()

        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in words
        ]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join(
            self.id_to_word.get(idx, self.unk_token)
            for idx in ids
        )