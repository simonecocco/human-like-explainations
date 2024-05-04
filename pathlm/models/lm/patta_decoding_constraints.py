from transformers import LogitsProcessor
from string import ascii_uppercase
import math
import torch
from pathlm.datasets import fill_template
from pathlm.models.lm.lm_utils import get_user_negatives_and_tokens_ids as token_ids_of_new_products
# devo importare una funzione che possa validarmi un'espressione o meno
import regex as re

# https://huggingface.co/docs/transformers/v4.40.1/en/internal/generation_utils#transformers.LogitsProcessor
class PattaLogitsProcessor(LogitsProcessor):
    __slots__: list[str] = [
        'max_output_len',
        'tokenizer_obj',
        'start_special_tokens',
        'end_special_tokens',
        'penality_dict',
        'product_regex',
        'relation_regex',
        'shared_entity_regex',
        'type_of_entity_regex',
    ]

    def __init__(self, max_len: int, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.max_output_len: int = max_len
        self.tokenizer_obj = tokenizer
        self.penality_dict: dict = {}
        self.product_regex: re.Pattern = re.compile(r'^P\d{1,}')
        self.relation_regex: re.Pattern = re.compile(r'^R[\d-]{1,}')
        self.shared_entity_regex: re.Pattern = re.compile(r'^[UE]\d{1,}')
        self.type_of_entity_regex: re.Pattern = re.compile(r'[A-Z]{1,}')
        # TODO per il finale è da
        # 1. elimina i PI che non portano a nulla
        # 2. aggiorna dinamicamente le relazioni e prodotti in base a ciò che sta costruendo
        self.start_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids([
            '<start_pi>', # get_user_positives
            '<start_rp>', # get_user_negatives
            '<start_se>', # evalutate.py:298
            '<start_te>', # fill_template.py:get_type_of_entity
            '<start_re>', # mi serve un dizionario che ha come chiave un'entità (E\d...) e come valore una lista di relazioni che supporta (attore supporta STARRED_BY)
        ])
        self.end_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids([
            '<end_pi>',
            '<end_rp>',
            '<end_se>',
            '<end_te>',
            '<end_re>',
        ])
        self.close_token: int = -1

    def compute_scores(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        def lock_on_token(token_id: int) -> None:
            scores[:, :] = -math.inf
            scores[:, token_id] = 0.0

        def elevate_only_subset_of_token(token_ids: list[int]) -> None:
            for i in range(scores.shape[-1]):
                if i not in token_ids:
                    scores[:, i] = -math.inf

        def deactivate_tokens(token_ids: list[int|str]) -> None:
            if type(token_ids[0]) == str:
                token_ids = self.tokenizer_obj.convert_tokens_to_ids(token_ids)
            scores[:, token_ids] = -math.inf

        def activate_if(rule: callable) -> None:
            for i in range(scores.shape[-1]):
                if not rule(i, self.tokenizer_obj.decode(i)):
                    scores[:, i] = -math.inf

        def apply_penality() -> None:
            for token_id in range(scores.shape[-1]):
                token_id = int(token_id)
                scores[:, token_id] -= math.exp(self.penality_dict.get(token_id, 0)) # TODO da valutare la funzione di penalità

        def update_penality() -> None:
            token_id = int(input_ids[0][-1])
            self.penality_dict.update({
                token_id:(self.penality_dict.get(token_id, 0) + 1)
            })
            for element in self.penality_dict.keys():
                if element != token_id:
                    self.penality_dict[element] -= 0.01 # TODO non superare lo 0 in positivo

        actual_token_len: int = input_ids.shape[-1] -1
        current_token_id: int = input_ids[0][-1]
        #print(input_ids.shape, current_token_id, self.tokenizer_obj.decode(current_token_id))

        update_penality()
        apply_penality()

        if actual_token_len == 4:
            activate_if(lambda i, token: self.product_regex.fullmatch(token) is not None)
        elif actual_token_len == 5:
            lock_on_token(self.tokenizer_obj.convert_tokens_to_ids('<end_rec>'))
        elif actual_token_len == 6:
            lock_on_token(self.tokenizer_obj.convert_tokens_to_ids('<start_exp>'))
        elif actual_token_len == self.max_output_len - 2:
            lock_on_token(self.tokenizer_obj.convert_tokens_to_ids('<end_exp>'))
        elif actual_token_len == self.max_output_len - 1 or current_token_id == self.tokenizer_obj.encode('<end_exp>')[0]:
            lock_on_token(self.tokenizer_obj.eos_token_id)
        elif current_token_id in self.start_special_tokens:
            self.close_token = self.start_special_tokens.index(current_token_id)
            # TODO usa dizionari al posto delle regex
            if current_token_id == self.start_special_tokens[0]:
                activate_if(lambda i, token: self.product_regex.fullmatch(token) is not None)
            elif current_token_id == self.start_special_tokens[1]:
                activate_if(lambda i, token: self.relation_regex.fullmatch(token) is not None)
            elif current_token_id == self.start_special_tokens[2]:
                activate_if(lambda i, token: self.shared_entity_regex.fullmatch(token) is not None)
            elif current_token_id == self.start_special_tokens[3]:
                activate_if(lambda i, token: self.type_of_entity_regex.fullmatch(token) is not None)
            elif current_token_id == self.start_special_tokens[4]:
                activate_if(lambda i, token: self.relation_regex.fullmatch(token) is not None)
        elif self.close_token != -1:
            lock_on_token(self.end_special_tokens[self.close_token])
            self.close_token = -1
        else:
            deactivate_tokens(['<end_rec>', '<start_exp>', '<end_exp>'])
            deactivate_tokens(self.end_special_tokens)
            activate_if(lambda i, token: token[0] not in ['E', 'P', 'R', 'U'])

        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.compute_scores(input_ids, scores)
