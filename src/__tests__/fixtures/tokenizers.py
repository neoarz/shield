"""A stand-in for the tokenizers package: one token per whitespace-separated word."""


class Encoding:
    def __init__(self, ids):
        self.ids = ids


class Tokenizer:
    @staticmethod
    def from_file(path):
        return Tokenizer()

    def no_padding(self):
        pass

    def no_truncation(self):
        pass

    def encode(self, text, add_special_tokens=True):
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            # The native binding refuses a lone surrogate this way.
            raise TypeError("TextInputSequence must be str") from None
        return Encoding(list(range(len(text.split()))))
