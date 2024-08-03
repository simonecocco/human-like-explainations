from transformers import LogitsProcessor
from math import exp, inf
import torch
from pathlm.datasets import fill_template
from pathlm.models.lm.tokenize_dataset import expand_patta_special_token
from pathlm.models.lm.lm_utils import get_user_negatives, get_user_positives
import regex as re
from pathlm.datasets import data_utils
from pathlm.utils import Cache
from pathlm.datasets.fill_template import fill_entities_ids, get_type_of_entity
from json import load

class PenalityLogitsProcessor(LogitsProcessor):
    __slots__: list[str] = [
        'penality_dict',
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.penality_dict: dict = {}

    def update_scores(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        for (index, val) in self.penality_dict.items():
            scores[:, index] -= exp(val)

        return scores

    def update_penality(self, input_ids: torch.LongTensor) -> None:
        last_id: int = int(input_ids[-1])
        self.penality_dict[last_id] = self.penality_dict.get(last_id, 1) + 1


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.update_penality(input_ids)
        scores = self.update_scores(scores)
        return scores

class BannedTokensLogitsProcessor(LogitsProcessor):
    __slots__: list[str] = [
        'tokenizer',
        'blacklist_token_map'
    ]

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.blacklist_token_map: list = []

    def add_token(self, token: str|int):
        if type(token) == str:
            token = self.tokenizer.convert_tokens_to_ids(token)
        self.blacklist_token_map.append(token)

    def remove_token(self, token: str|int):
        if type(token) == str:
            token = self.tokenizer.convert_tokens_to_ids(token)
        del self.blacklist_token_map[self.blacklist_token_map.index(token)]

    def ban_tokens(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token in self.blacklist_token_map:
            scores[:, token] = -inf
        return scores
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.ban_tokens(scores)


class PattaLogitsProcessor(LogitsProcessor):
    __slots__: list[str] = [
        'persistent_cache_storage',
        'cache_dict'
    ]

    def __init__(self, max_len: int, tokenizer, dataset_name, banned_tokens_lp, **kwargs):
        super().__init__(**kwargs)
        
        self.persistent_cache_storage: Cache = Cache()
        self.cache_dict: dict = {}

        entities_ids_name = f'entities_{dataset_name}'
        if not (entities_ids_name in self.persistent_cache_storage):
            entities_ids = fill_entities_ids(dataset_name)
            
            self.cache_dict[entities_ids_name] = entities_ids
            self.persistent_cache_storage[entities_ids_name] = entities_ids
        else:
            entities_ids = self.persistent_cache_storage[entities_ids_name]
            self.cache_dict[entities_ids_name] = entities_ids

        

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

        self.load_big_data(f'{dataset_name}_user_positives', lambda: )
        self.shared_entity_regex: re.Pattern = re.compile(r'^[UE]\d{1,}')
        self.type_of_entity_regex: re.Pattern = re.compile(r'[A-Z]{1,}')
        self.list_of_tokens = self.tokenizer_obj.get_vocab()

        self.start_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids(expand_patta_special_token(with_prefix=['start']))
        self.end_special_tokens: list[int] = self.tokenizer_obj.convert_tokens_to_ids(expand_patta_special_token(with_prefix=['end']))
        self.close_token: int = -1


    def load_big_data(self, data_name: str, data_loader: callable, alias: str=None) -> None:
        if not (data_name in self.persistent_cache_storage):
            data = data_loader()
            self.cache_dict[data_name if alias is None else alias] = data
            self.persistent_cache_storage[data_name] = data
        else:
            self.cache_dict[data_name if alias is None else alias] = self.persistent_cache_storage[data_name]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores
