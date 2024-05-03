from transformers import Trainer
from pathlm.models.lm.patta_decoding_constraints import PattaLogitsProcessorBPE

class PattaTrainer(Trainer):
    __slots__: list[str] = [
        'logits_processor'
    ]

    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.logits_processor = PattaLogitsProcessorBPE(256, tokenizer)
        # TODO