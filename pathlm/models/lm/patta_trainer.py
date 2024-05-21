from transformers import Trainer

class PattaTrainer(Trainer):
    __slots__: list[str] = []

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)