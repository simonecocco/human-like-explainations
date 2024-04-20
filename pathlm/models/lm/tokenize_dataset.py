import argparse
import os
from os.path import join
from typing_extensions import deprecated
from datasets import DatasetDict
from responses import remove
from sympy import Union
from tokenizers import (
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    AddedToken
)

from typing import Union
from transformers import PreTrainedTokenizerFast, set_seed, AutoTokenizer, PreTrainedTokenizer

from build.lib.pathlm.utils import get_raw_paths_dir
from pathlm.models.lm.path_dataset import PathDataset
from pathlm.sampling import KGsampler
from pathlm.utils import *
import pandas as pd
from datasets import Dataset

PATTA_LM: dict = {
    'tokenizer_type': 'BPE',
    'special_tokens': {
        'start_pi_token':'<start_pi>',
        'end_pi_token':'<end_pi>',
        'start_rp_token':'<start_rp>',
        'end_rp_token':'<end_rp>',
        'start_se_token':'<start_se>',
        'end_se_token':'<end_se>',
        'start_te_token':'<start_te>',
        'end_te_token':'<end_te>',
        'start_re_token':'<start_re>',
        'end_re_token':'<end_re>',
        'start_rec_token':'<start_rec>',
        'end_rec_token':'<end_rec>',
        'start_exp_token':'<start_exp>',
        'end_exp_token':'<end_exp>',
    }
}

@deprecated('This function is deprecated. Use get_tokenize_function instead')
def tokenize_function(examples: str, context_length: int=200):
    return tokenizer(examples["path"], truncation=True, padding=True, max_length=context_length) # type: ignore

def get_tokenize_function(tokenizer, context_length: int=-1):
    return lambda examples: tokenizer(examples['path'], padding=True, truncation=True, max_length=512)

def extend_tokenizer(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], words: set) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    print(f'Adding {len(PATTA_LM["special_tokens"])} special tokens')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # TODO !!
    special_tokens: list[AddedToken] = [
        AddedToken(token, single_word=True, lstrip=True, rstrip=True, normalized=False)
        for token in PATTA_LM["special_tokens"]
    ]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens}) # type: ignore
    print(f'Adding {len(words)} words')
    normal_words: list[AddedToken] = [
        AddedToken(word, single_word=True, lstrip=True, rstrip=True, normalized=False)
        for word in words
    ]
    tokenizer.add_tokens(normal_words) # type: ignore
    #tokenizer.resize_token_embeddings(len(special_tokens) + len(normal_words))
    print('Done')

    return tokenizer

def expand_and_save_patta_tokenizer(model: str, word_list: set, tokenizer_file_path: str):
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = AutoTokenizer.from_pretrained(model)
    tokenizer = extend_tokenizer(tokenizer, word_list)
    print(f'Saving tokenizer to {tokenizer_file_path}')
    tokenizer.save_pretrained(tokenizer_file_path)
    return tokenizer

def tokenize_dataset_for_patta_lm(tokenizer, dataset, data_output_dir: str, num_proc: int=4):
    print('Tokenizing dataset...')
    tokenized_dataset = dataset.map(
        get_tokenize_function(tokenizer, context_length=tokenizer.model_max_length),
        batched=True,
        num_proc=num_proc,
        remove_columns=["path"]
    )
    tokenized_dataset = DatasetDict({'train': tokenized_dataset})
    tokenized_dataset.save_to_disk(join(data_output_dir, 'tokenized_dataset.hf'))

def tokenize_for_patta_lm(args: argparse.Namespace):
    tokenizer_file_path: str = join(get_tokenizer_dir_path(args.dataset, args.model, 'patta'), 'tokenizer')
    raw_paths_file_path: str = join(get_raw_paths_dir(args.dataset), args.raw_paths_file_name)
    filled_templates_file_path: str = join(get_filled_templates_dir(args.dataset), args.filled_templates_file_name)
    
    with open(raw_paths_file_path) as f:
        word_list: set = {
            word.strip()
            for row in f.read().strip().split('\n')
            for word in row.strip().split(' ')
        }

    tokenizer = expand_and_save_patta_tokenizer(args.model, word_list, tokenizer_file_path)
    df = pd.read_csv(filled_templates_file_path, header=None, names=["path"], index_col=None, sep='\t')
    dataset = Dataset.from_pandas(df)
    print(dataset)
    tokenized_dataset_dir_path: str = get_tokenized_dataset_dir_path(args.dataset, args.model, 'patta')
    tokenize_dataset_for_patta_lm(tokenizer, dataset, tokenized_dataset_dir_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--dataset", type=str, default="ml1m", choices=['ml1m', 'lfm1m'], help="Dataset to use")
    parser.add_argument("--task", type=str, default="end-to-end", choices=['pretrain', 'end-to-end'])
    parser.add_argument("--sample_size", type=str, default="250",
                        help="Number of sampled path in the chosen dataset")
    parser.add_argument("--n_hop", type=int, default=3,
                        help="Number of elements in a predicted sequence (considering only the ids)")
    parser.add_argument("--context_length", type=int, default=24,
                        help="Context length value when training a tokenizer from scratch")
    parser.add_argument("--nproc", type=int, default=8, help="Number of processes for dataset mapping")

    # Patta
    parser.add_argument('-iFT', '--filled-templates-file-name', default='filled_template.txt', type=str, help='Path to the filled template file')
    parser.add_argument('-iRP' ,'--raw-paths-file-name', default='paths_end-to-end_250_3.txt', type=str, help='Path to load paths')
    parser.add_argument('-M', '--model', default='distilgpt2', type=str, help='Model name from Hugging Face')
    parser.add_argument('--only-patta', action='store_true', help='Tokenize only for Patta LM')

    args = parser.parse_args()

    set_seed(SEED)

    tokenize_for_patta_lm(args)

    if args.only_patta:
        exit(0)

    dataset_root_dir = get_root_data_dir(args.dataset)
    args.tokenizer_dir = './tokenizers'
    TOKENIZER_TYPE = "WordLevel"
    dataset_name = args.dataset

    tokenizer_dir = os.path.join(args.tokenizer_dir, dataset_name)
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer_file = os.path.join(tokenizer_dir, f"{TOKENIZER_TYPE}.json")

    dirpath = get_data_dir(dataset_name)
    data_dir_mapping = os.path.join(dirpath, f'mapping/')
    kg = KGsampler(args, args.dataset, dirpath)
    sample_size = args.sample_size
    dataset_hop_size = args.n_hop
    TOKENIZED_DATASET_PATH = os.path.join(dataset_root_dir, f"{TOKENIZER_TYPE}/{args.task}_{sample_size}_{dataset_hop_size}_tokenized_dataset.hf")
    TOKEN_INDEX_PATH = os.path.join(dirpath, KGsampler.TOKEN_INDEX_FILE)
    # Try to load the dataset from disk if it has been already tokenized otherwise load it from scratch
    plain_text_path = True

    print("Loading and processing path sequences...")
    dataset = PathDataset(dataset_name, dataset_root_dir, task=args.task, sample_size=sample_size, n_hop=dataset_hop_size,
                          plain_text_path=plain_text_path)

    dataset.show_random_examples()
    dataset = dataset.dataset
    print(type(dataset))

    # Word level tokenizer
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]")) # type: ignore
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit() # type: ignore
    trainer = trainers.WordLevelTrainer(special_tokens=special_tokens) # type: ignore

    tokens = []
    with open(TOKEN_INDEX_PATH) as f:
        for line in f:
            tokens.append(line.rstrip())
    tokenizer.train_from_iterator(tokens, #dataset["path"],
                                    trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS]:0 $A:0 [EOS]:0",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")), ("[EOS]", tokenizer.token_to_id("[EOS]"))]
    ) # type: ignore

    tokenizer.save(tokenizer_file)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, max_len=args.context_length,
                                        eos_token="[EOS]", bos_token="[BOS]",
                                        pad_token="[PAD]", unk_token="[UNK]",
                                        mask_token="[MASK]", use_fast=True)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function,
                                    batched=True,
                                    num_proc=args.nproc,
                                    remove_columns=["path"]
                                    )
    tokenized_dataset = DatasetDict({
        "train": tokenized_dataset,
    })
    # Create a dir if does not exist for the hf dataset and save the tokenized dataset to disk
    check_dir(TOKENIZED_DATASET_PATH)
    tokenized_dataset.save_to_disk(
        TOKENIZED_DATASET_PATH)
