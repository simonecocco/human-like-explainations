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
        metric_for_best_model='accuracy',
        greater_is_better=True,
        save_strategy='epoch',
        eval_accumulation_steps=10
    )

def compute_metrics(pred):
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
        #compute_metrics=lambda pred: {'accuracy': torch.sum(pred.label_ids == pred.predictions.argmax(-1)).item()}
        #sol1
        #compute_metrics=lambda pred: {'accuracy': torch.sum(torch.from_numpy(pred.label_ids) == pred.predictions.argmax(-1)).item()}
        #sol2
        #compute_metrics=lambda pred: {'accuracy': torch.sum(torch.tensor(pred.label_ids) == pred.predictions.argmax(-1)).item()}
        #sol3
        #compute_metrics=lambda pred: {'accuracy': torch.sum(torch.where(torch.tensor(pred.label_ids) == pred.predictions.argmax(-1), torch.tensor(1), torch.tensor(0))).item()}
        #sol4
        compute_metrics=compute_metrics
    )

def train_patta_lm(args, tokenizer, tokenized_dataset):
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    tokenized_dataset_data_loader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)
    trainer = get_trainer_obj(args, tokenized_dataset['train'], model)
    trainer.train()

    weight_path = get_weight_dir(f'patta_{args.epochs}', args.dataset)
    trainer.save_model(weight_path)

if __name__ == '__main__':
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml1m', choices=['ml1m', 'lfm1m'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='distilgpt2',
                        help='Model name from Hugging Face')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train the model')
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
    #tokenized_dataset = tokenized_dataset.map(lambda x: )
    train_patta_lm(args, tokenizer, tokenized_dataset)