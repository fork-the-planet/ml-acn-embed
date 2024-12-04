# pylint: disable=duplicate-code
class WordTable:
    def __init__(self, filename):
        self.id_to_word = {}
        self.word_to_id = {}
        with open(filename, "r", encoding="utf8") as fobj:
            for line in fobj:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                word = tokens[0]
                num = int(tokens[1])
                self.id_to_word[num] = word
                self.word_to_id[word] = num
