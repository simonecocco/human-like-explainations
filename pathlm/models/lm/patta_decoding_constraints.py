from sympy import expand
from transformers import LogitsProcessor
from string import ascii_uppercase
import math
import torch
from pathlm.datasets import fill_template
from pathlm.models.lm.tokenize_dataset import expand_patta_special_token
from pathlm.models.lm.lm_utils import get_user_negatives_and_tokens_ids as token_ids_of_new_products
import regex as re
from pathlm.datasets import data_utils
from pathlm.utils import Cache
from pathlm.datasets.fill_template import fill_entities_ids, get_type_of_entity
from json import load

# https://huggingface.co/docs/transformers/v4.40.1/en/internal/generation_utils#transformers.LogitsProcessor
class PattaLogitsProcessor(LogitsProcessor):
    __slots__: list[str] = [
        'special_token_behavior',
        'banned_token_ids'
    ]

    def __init__(self, max_len: int, tokenizer, dataset_name, **kwargs):
        super().__init__(**kwargs)
        
        self.entities_ids = fill_entities_ids(dataset_name)
        self.special_already_used_tokens: list[int] = [tokenizer.convert_tokens_to_ids('<end_exp>')]
        self.max_output_len: int = max_len
        self.penality_tensor = None
        self.cache: dict = {}
        self.tokenizer_obj = tokenizer
        self.penality_dict: dict = {}
        self.persistent_cache: Cache = Cache()
        if not (f'{dataset_name}_user_positives' in self.persistent_cache):
            self.store_cache('user_positives', data_utils.get_user_positives(dataset_name))
            self.store_cache('user_negatives', data_utils.get_user_negatives(dataset_name))
            self.persistent_cache[f'{dataset_name}_user_positives'] = self.get_cache('user_positives')
            self.persistent_cache[f'{dataset_name}_user_negatives'] = self.get_cache('user_negatives')
        else:
            self.store_cache('user_positives', self.persistent_cache[f'{dataset_name}_user_positives'])
            self.store_cache('user_negatives', self.persistent_cache[f'{dataset_name}_user_negatives'])
        self.shared_entity_regex: re.Pattern = re.compile(r'^[UE]\d{1,}')
        self.type_of_entity_regex: re.Pattern = re.compile(r'[A-Z]{1,}')
        self.list_of_tokens = self.tokenizer_obj.get_vocab()

        self.start_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids(expand_patta_special_token(with_prefix=['start']))
        self.end_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids(expand_patta_special_token(with_prefix=['end']))
        self.close_token: int = -1

    def edit_logits(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        last_token_id: int = int(input_ids[0][-1])
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.edit_logits(input_ids, scores)
