import heapq
import logging
import multiprocessing
import os
import random

import numpy as np
import torch

from pathlm.datasets.data_utils import get_user_negatives
from pathlm.evaluation.eval_metrics import evaluate_rec_quality

from pathlm.models.knowledge_aware.KGAT.batch_test import evaluate_model
from pathlm.models.knowledge_aware.KGAT.dataloader_kgat import KGATLoader
from pathlm.models.knowledge_aware.KGAT.parser_kgat import parse_args
from pathlm.models.model_utils import EarlyStopping, logging_metrics
from pathlm.models.traditional.log_helper import logging_config, create_log_id
from pathlm.utils import SEED, get_model_data_dir
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from time import time
from KGAT import KGAT


def initialize_model(args, dataset_obj):
    config = {
        'n_users': dataset_obj.n_users,
        'n_items': dataset_obj.n_items,
        'n_relations': dataset_obj.n_relations,
        'n_entities': dataset_obj.n_entities,
        'A_in': sum(dataset_obj.lap_list),
        'all_h_list': dataset_obj.all_h_list,
        'all_r_list': dataset_obj.all_r_list,
        'all_t_list': dataset_obj.all_t_list,
        'all_v_list': dataset_obj.all_v_list,
        'n_relations': dataset_obj.n_relations,
    }
    model = KGAT(config, pretrain_data=None, args=args).to(args.device)
    return model

#Note: the original model, samples uniformely without replacement inside a batch but multiple batches with respect to one
#epoch can contain the same datapoints, this can make the learning unstable or enforce biases (that may cause overfit and
#consequently better ndcg)
def get_data_loader(dataset_obj, args, mode, n_proc=16):
    dataset_obj.mode = mode
    batch_size = dataset_obj.batch_size if mode == 'cf' else dataset_obj.batch_size_kg
    sampler = RandomSampler(dataset_obj, replacement=args.with_replacement, generator=torch.Generator().manual_seed(SEED)) if args.with_replacement else None
    #shuffle = not args.with_replacement
    data_loader = DataLoader(dataset_obj,
                             batch_size=batch_size,
                             sampler=sampler,
                             shuffle=True,
                             num_workers=n_proc,
                             drop_last=True,
                             persistent_workers=False)
    return data_loader

def train_epoch(model, data_loader, mode, epoch, args):
    total_loss, total_base_loss, total_kge_loss, total_reg_loss = 0.0, 0.0, 0.0, 0.0
    for iter, batch_data in enumerate(data_loader, 1):
        batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train_step(batch_data, mode=mode)
        total_loss += batch_loss.item()
        total_base_loss += batch_base_loss.item()
        total_kge_loss += batch_kge_loss.item()
        total_reg_loss += batch_reg_loss.item()
        if (iter % args.print_every) == 0:
            logging.info(f'CF Training: Epoch {epoch:04d} Iter {iter:04d} / {len(data_loader):04d} | '
                         f'Iter Loss {batch_loss.item():.4f} | '
                         f'Iter Mean Loss {total_loss / iter:.4f}')
    return total_loss / len(data_loader), total_base_loss / len(data_loader), total_kge_loss, total_reg_loss / len(data_loader)

def train_epoch_kge(model, data_generator, mode, epoch, args):
    total_loss, total_base_loss, total_kge_loss, total_reg_loss = 0.0, 0.0, 0.0, 0.0
    n_batch = len(data_generator.all_h_list) // args.batch_size_kg + 1
    for iter in range(1, n_batch):
        batch_data = data_generator._generate_kge_train_batch()
        batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = model.train_step(batch_data, mode=mode)
        total_loss += batch_loss.item()
        total_base_loss += batch_base_loss.item()
        total_kge_loss += batch_kge_loss.item()
        total_reg_loss += batch_reg_loss.item()
        if (iter % args.print_every) == 0:
            logging.info(f'KG Training: Epoch {epoch:04d} Iter {iter:04d} / {n_batch:04d} | '
                         f'Iter Loss {batch_loss.item():.4f} | '
                         f'Iter Mean Loss {total_loss / iter:.4f}')
    return total_loss, total_base_loss, total_kge_loss / n_batch, total_reg_loss / n_batch

def print_training_info(epoch, train_time, avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss):
    logging.info(f"Epoch {epoch} [{train_time:.1f}s]: Average loss: {avg_loss:.5f} = "
          f"Base loss: {avg_base_loss:.5f} + KGE loss: {avg_kge_loss:.5f} + Reg loss: {avg_reg_loss:.5f}")


def ranklist_by_heapq(user_negatives, rating, K):
    item_score = {i: rating[i] for i in user_negatives}
    K_max_item_score = heapq.nlargest(K, item_score, key=item_score.get)
    return K_max_item_score

def evaluate_model(model, users_to_test, kgat_dataset, args):
    K = args.K
    u_batch_size = args.test_batch_size * 2
    n_test_users = len(users_to_test)
    n_user_batches = n_test_users // u_batch_size + 1
    topks = {}
    user_negatives = get_user_negatives(args.dataset)
    model.eval()  # Set the model to evaluation mode

    for u_batch_id in range(n_user_batches):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        user_batch = users_to_test[start: end]
        item_batch = list(range(0, model.n_items+1))
        feed_dict = kgat_dataset.prepare_test_data(user_batch, item_batch)
        # Forward pass through the model
        rate_batch = model(feed_dict, 'eval')  # Assuming model's forward pass returns the required ratings.
        rate_batch = rate_batch.detach().cpu().numpy()  # Convert to numpy for subsequent operations

        for i, user in enumerate(user_batch):
            user_ratings = rate_batch[i]
            candidate_items = user_negatives[user]
            pids = ranklist_by_heapq(candidate_items, user_ratings, K)
            topks[user] = pids

    avg_metrics_dict = evaluate_rec_quality(args.dataset, topks, kgat_dataset.test_user_dict, K)[1]

    return avg_metrics_dict, topks

def train(args):
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Setup logging
    log_save_id = create_log_id(args.log_dir)
    logging_config(folder=args.log_dir, name=f'log{log_save_id}', no_console=False)
    logging.info(args)

    # Setup device (GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    train_cores = multiprocessing.cpu_count()

    dataset_obj = KGATLoader(args=args, path=get_model_data_dir(args.model_type, args.dataset))
    cf_data_loader = get_data_loader(dataset_obj, args, mode='cf', n_proc=train_cores)
    #kge_data_loader = get_data_loader(dataset_obj, args, mode='kg')

    model = initialize_model(args, dataset_obj)
    logging.info(model)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in tqdm(range(args.epoch)):
        t1 = time()

        # Phase 1: CF training
        cf_data_loader.dataset.set_mode('cf')
        avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss = train_epoch(model, cf_data_loader, 'rec', epoch, args)
        #avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss = train_epoch(model, dataset_obj, 'rec', args)
        print_training_info(epoch, time() - t1, avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss)
        assert np.isnan(avg_loss) == False
        #avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss = 0.0,0.0,0.0,0.0
        # Phase 2: KGE training and update attentive Laplacian matrix
        if args.use_kge:
            #kge_data_loader.dataset.set_mode('kg')
            #_, _, avg_kge_loss, avg_reg_loss = train_epoch(model, kge_data_loader, 'kge', args)
            _, _, avg_kge_loss, avg_reg_loss = train_epoch_kge(model, dataset_obj, 'kge', epoch, args)
            avg_loss += avg_kge_loss
            print_training_info(epoch, time() - t1, avg_loss, avg_base_loss, avg_kge_loss, avg_reg_loss)

            if args.use_att:
                model.update_attentive_A()

            print(f"KGE Training completed. Average KGE loss: {avg_kge_loss:.5f}")
        assert np.isnan(avg_kge_loss) == False

        # Phase 3: Test
        # Testing and performance logging
        t2 = time()
        users_to_test = list(dataset_obj.test_user_dict.keys())
        test_metrics, topks = evaluate_model(model, users_to_test, dataset_obj, args)
        logging_metrics(epoch, test_metrics, [str(args.K)])

        ndcg_value = test_metrics['ndcg']
        early_stopping(ndcg_value)

        if early_stopping.early_stop:
            logging.info('Early stopping triggered. Stopping training.')
            break

        # Optional: Save model and metrics at each epoch or at specific intervals
        if epoch % args.save_interval == 0 or epoch == args.epoch - 1:
            torch.save(model.state_dict(), os.path.join(args.weight_dir_ckpt, f'{args.model_type}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
        # Final model save and cleanup
    torch.save(model.state_dict(), os.path.join(args.weight_dir, f'{args.model_type}_epoch_{epoch}_e{args.embed_size}_bs{args.batch_size}_lr{args.lr}.pth'))
    logging.info(f'Best evaluation results at epoch {early_stopping.best_epoch} with NDCG: {early_stopping.best_score:.4f}')

if __name__ == '__main__':
    args = parse_args()
    train(args)
