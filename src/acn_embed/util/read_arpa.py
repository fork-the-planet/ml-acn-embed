import gzip


def is_numeric(string):
    try:
        float(string)
    except ValueError:
        return False
    return True


def get_ngrams(arpa_fn) -> set:
    ngrams = set()
    state = 0
    with gzip.open(arpa_fn, "rt") as fobj:
        for line in fobj:
            line = line.strip()
            if not line:
                continue
            if line.startswith("\\end\\"):
                break
            if line.startswith("\\") and line.endswith("-grams:"):
                state = 1
                continue
            if state == 0:
                continue
            tokens = line.split()
            assert len(tokens) >= 2
            if len(tokens) == 2:
                words = tokens[1:2]
            else:
                if is_numeric(tokens[-1]):
                    words = tokens[1:-1]
                else:
                    words = tokens[1:]
            # pylint: disable=use-a-generator
            if any([word.lower() in ["<unk>", "<s>", "</s>"] for word in words]):
                continue
            ngrams.add(" ".join(words))
    return ngrams
