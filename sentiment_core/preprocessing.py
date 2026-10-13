from  collections import Counter

class Vocabulary:
    """
    Maps words to integer indices and back. 
    Built from a training corpus
    reused at inference time to encode new text consistently
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    MAX_LEN = 30
    MIN_FREQ = 2

    def __init__(self) -> None: 
        self.word_to_idx: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.idx_to_word: dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }

    def build(self, texts: list[str], min_freq: int = MIN_FREQ) -> None:
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split()) # tokenization

        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word

    def encode(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        tokens = text.lower().split()
        ids = [self.word_to_idx.get(t, self.word_to_idx[self.UNK_TOKEN]) for t in tokens]
        ids = ids[:max_len]
        ids += [self.word_to_idx[self.PAD_TOKEN]] * (max_len - len(ids)) # Python list multiplication 
        return ids

    def __len__(self)-> int:
        return len(self.word_to_idx)