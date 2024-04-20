from argparse import ArgumentParser
from pathlm.utils import *
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from os.path import exists
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
from pathlm.models.lm.tokenize_dataset import PATTA_LM

def get_training_args_obj(args):
    return TrainingArguments(
        output_dir=get_checkpoint_dir_path('patta', args.model), # TODO cambia
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy='epoch',
        learning_rate=5e-5,
        save_total_limit=2,
        save_steps=500,
        load_best_model_at_end=True,
        #metric_for_best_model='accuracy',
        greater_is_better=True,
        save_strategy='epoch',
        eval_accumulation_steps=10
    )

def compute_metrics(pred):
    # https://huggingface.co/docs/transformers/v4.40.0/en/internal/trainer_utils#transformers.EvalPrediction
    loss = pred.loss
    perplexity = torch.exp(loss)
    return {'perplexity': perplexity.item(), 'loss': loss.item()}

def get_trainer_obj(args, tokenized_dataset, model):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.15
    )
    return Trainer(
        model=model,
        args=get_training_args_obj(args),
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        eval_dataset=tokenized_dataset,
    )

def train_patta_lm(args, tokenizer, tokenized_dataset):
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    trainer = get_trainer_obj(args, tokenized_dataset['train'], model)
    trainer.train()
    print('Model trained')

    weight_path = get_weight_dir(f'patta_{args.model}', args.dataset)
    trainer.save_model(weight_path)
    print(f'Model saved at {weight_path}')

    return model

if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1m', choices=['ml1m', 'lfm1m'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='distilgpt2',
                        help='Model name from Hugging Face')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train the model')
    parser.add_argument('--eval', type=str,
                        help='If the model is already trained, evaluate it sending a path')
    args = parser.parse_args()

    set_seed(SEED)

    tokenized_dataset_path: str = join(get_tokenized_dataset_dir_path(args.dataset, args.model, 'patta'), 'tokenized_dataset.hf')
    if not exists(tokenized_dataset_path):
        print('Run the tokenization script first')
        exit(1)
    tokenized_dataset = load_from_disk(tokenized_dataset_path)
    print('Tokenized dataset loaded')
    tokenizer_file_path: str = join(get_tokenizer_dir_path(args.dataset, args.model, 'patta'), 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_file_path, max_length=512, padding='max_length', truncation=True)
    weight_path = get_weight_dir(f'patta_{args.model}', args.dataset)
    if exists(weight_path):
        model = AutoModelForCausalLM.from_pretrained(weight_path)
    else: 
        model = train_patta_lm(args, tokenizer, tokenized_dataset)

    # TODO la generazione di sequenze non funziona
    if args.eval is not None:
        sequence_to_generate = f"{PATTA_LM['special_tokens']['start_rec_token']} {args.eval} {PATTA_LM['special_tokens']['end_rec_token']} {PATTA_LM['special_tokens']['start_exp_token']}  {PATTA_LM['special_tokens']['end_exp_token']}"
        print(f'Generating sequence: {sequence_to_generate}')
        input_ids = tokenizer.encode(sequence_to_generate, return_tensors='pt')
        output = model(input_ids)
        token_ids = torch.argmax(output.logits, dim=2)
        generated_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        print(f'Generated text: {generated_text}')