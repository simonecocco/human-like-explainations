import csv
from genericpath import exists
from os.path import join
import os
from typing import Dict
import gzip

SEED = 2023

import torch 
import random
import numpy as np

def check_dir(dir_path: str) -> None:
    """
    Check if directory exists and create it if not
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def normalise_name(name: str) -> str:
    """
    Clean entities name from _ or previxes
    """
    if name.startswith("Category:"):
        name = name.replace("Category:", "")
    return name.replace("_", " ")

def set_seed(seed=SEED, use_deterministic=True):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(use_deterministic)
    np.random.seed(seed) 
    random.seed(seed)

def get_dataset_id2eid(dataset_name: str, what: str="user") -> Dict[str, str]:
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    file = open(os.path.join(data_dir, f"mapping/{what}.txt"), "r")
    csv_reader = csv.reader(file, delimiter='\t')
    dataset_pid2eid = {}
    next(csv_reader, None)
    for row in csv_reader:
        dataset_pid2eid[row[1]] = row[0]
    file.close()
    return dataset_pid2eid

def get_eid2dataset_id(dataset_name: str, what: str="user") -> Dict[str, str]:
    data_dir = os.path.join('data', dataset_name, 'preprocessed')
    file = open(os.path.join(data_dir, f"mapping/{what}.txt"), "r")
    csv_reader = csv.reader(file, delimiter='\t')
    eid2dataset_id = {}
    next(csv_reader, None)
    for row in csv_reader:
        eid2dataset_id[row[0]] = row[1]
    file.close()
    return eid2dataset_id

def get_rid_to_name_map(dataset_name: str) -> dict:
    """
    Get rid2name dictionary to allow conversion from rid to name
    """
    r_map_path = join(f'data/{dataset_name}', 'preprocessed/r_map.txt')
    rid2name = {}
    with open(r_map_path) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            rid = row[0]
            rname = normalise_name(row[-1])
            rid2name[rid] = rname
    f.close()
    return rid2name

def get_data_dir(dataset_name: str) -> str:
    return join('data', dataset_name, 'preprocessed')

def get_root_data_dir(dataset_name: str) -> str:
    return join('data', dataset_name)

def get_data_template_dir(dataset_name: str) -> str:
    data_template_dir_path: str = join(get_root_data_dir(dataset_name), 'templates')
    check_dir(data_template_dir_path)
    return data_template_dir_path

def get_model_data_dir(model_name: str, dataset_name: str) -> str:
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    return join(get_data_dir(dataset_name), model_name)

def get_weight_dir(model_name: str, dataset_name: str) -> str:
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    weight_dir_path = join('weights', dataset_name, model_name)
    check_dir(weight_dir_path)
    return weight_dir_path

def get_weight_ckpt_dir(model_name: str, dataset_name: str) -> str:
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    weight_ckpt_dir_path = join(get_weight_dir(model_name, dataset_name), 'ckpt')
    check_dir(weight_ckpt_dir_path)
    return weight_ckpt_dir_path


def get_eid_to_name_map(dataset_name: str) -> Dict[str, str]:
    eid2name = dict()
    with open(os.path.join(f'data/{dataset_name}/preprocessed/e_map.txt')) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            eid, name = row[:2]
            eid2name[eid] = ' '.join(name.split('_'))
    return eid2name

def get_raw_paths_dir(dataset_name: str) -> str:
    raw_paths_dir_path = join(get_root_data_dir(dataset_name), 'paths_random_walk')
    check_dir(raw_paths_dir_path)
    return raw_paths_dir_path

def get_filled_templates_dir(dataset_name: str) -> str:
    filled_templates_path = join(get_root_data_dir(dataset_name), 'filled_templates')
    check_dir(filled_templates_path)
    return filled_templates_path

def get_entity_ids_file_path(dataset_name: str, what_type_of_entity: str) -> str:
    mapping_dir_path: str = join(get_root_data_dir(dataset_name), 'preprocessed/mapping')
    check_dir(mapping_dir_path)
    file_name: str = f'{what_type_of_entity}.txt'
    if exists(join(mapping_dir_path, file_name)):
        return join(mapping_dir_path, file_name)
    else:
        return join(mapping_dir_path, f'{what_type_of_entity}.txt.gz')

def read_entity_ids_file(entity_ids_file_path: str) -> list:
    entity_ids: dict = {}
    if entity_ids_file_path.endswith('.gz'):
        with gzip.open(entity_ids_file_path, 'rt') as f:
            reader = csv.reader(f, delimiter='\t')
            return [row[1] for row in reader]
    else:
        with open(entity_ids_file_path) as f:
            reader = csv.reader(f, delimiter='\t')
            return [row[1] for row in reader]

def get_tokenizer_dir_path(dataset_name: str, model_name: str, lm_name: str) -> str:
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    tokenizer_dir_path: str = join(get_root_data_dir(dataset_name), 'tokenizers')
    check_dir(tokenizer_dir_path)
    lm_tokenizer_dir_path: str = join(tokenizer_dir_path, lm_name)
    check_dir(lm_tokenizer_dir_path)
    model_tokenizer_dir_path: str = join(lm_tokenizer_dir_path, model_name)
    check_dir(model_tokenizer_dir_path)
    return model_tokenizer_dir_path

def get_tokenized_dataset_dir_path(dataset_name: str, model_name: str, lm_name: str) -> str:
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    tokenized_dataset_dir_path: str = join(get_tokenizer_dir_path(dataset_name, model_name, lm_name), 'tokenized_dataset')
    check_dir(tokenized_dataset_dir_path)
    return tokenized_dataset_dir_path

def get_checkpoint_dir_path(lm_name: str, model_name: str) -> str:
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    checkpoint_dir_path: str = join('checkpoints', lm_name, model_name)
    check_dir(checkpoint_dir_path)
    return checkpoint_dir_path

def get_model_weights_dir_path(lm_name: str, model_name: str) -> str:
    if '/' in model_name:
        model_name = model_name.replace('/', '_')
    model_weights_dir_path: str = join('model_weights', lm_name, model_name)
    check_dir(model_weights_dir_path)
    return model_weights_dir_path