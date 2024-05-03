from transformers import LogitsProcessor
import math
import torch
from pathlm.datasets import fill_template
from pathlm.models.lm.lm_utils import get_user_negatives_and_tokens_ids as token_ids_of_new_products

# https://huggingface.co/docs/transformers/v4.40.1/en/internal/generation_utils#transformers.LogitsProcessor
class PattaLogitsProcessorBPE(LogitsProcessor):
    __slots__: list[str] = [
        'max_output_len',
        'tokenizer_obj',
        'start_special_tokens',
        'end_special_tokens'
    ]

    def __init__(self, max_len: int, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.max_output_len: int = max_len
        self.tokenizer_obj = tokenizer
        self.start_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids([
            '<start_pi>',
            '<start_rp>',
            '<start_se>',
            '<start_te>',
            '<start_re>',
        ])
        self.end_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids([
            '<end_pi>',
            '<end_rp>',
            '<end_se>',
            '<end_te>',
            '<end_re>',
        ])
        self.close_token: int = -1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        def lock_on_token(token_id: int) -> None:
            scores[:, :] = -math.inf
            scores[:, token_id] = 0.0

        def elevate_only_subset_of_token(token_ids: list[int]) -> None:
            for i in range(scores.shape[-1]):
                if i not in token_ids:
                    scores[:, i] = -math.inf

        actual_token_len: int = input_ids.shape[-1]
        if actual_token_len == self.max_output_len - 1:
            # fermo la generazione dell'espressione
            lock_on_token(self.tokenizer_obj.convert_tokens_to_ids('<end_exp>'))
        elif actual_token_len == self.max_output_len:
            # fermo la generazione
            lock_on_token(self.tokenizer_obj.eos_token_id)
        # predici un prodotto
        #elif actual_token_len == 3\
        #    or input_ids[-1] in self.start_special_tokens[:-1]:
            # azzera i non prodotti
            # TODO
        #    pass
        #elif input_ids[-1] == self.start_special_tokens[-1]: # predici la relazione
            # TODO
        #    pass
        
        elif self.close_token != -1: # predici un close token
            lock_on_token(self.end_special_tokens[self.close_token])
            self.close_token = -1
        elif actual_token_len == 4: # predici <end_rec>
            lock_on_token(self.tokenizer_obj.convert_tokens_to_ids('<end_rec>'))
        elif actual_token_len == 5: # predici <start_exp>
            lock_on_token(self.tokenizer_obj.convert_tokens_to_ids('<start_exp>'))

        return scores