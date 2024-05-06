from time import time

import torch
import numpy as np
from .metrics import precision_at_k, recall_at_k, ndcg_at_k, hit_at_k


def test(model, user_dict, n_params, args):
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    Ks = eval(args.Ks)
    batch_size = args.test_batch_size
    n_items = n_params['n_items']
    n_users = n_params['n_users']
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    result = {
        'precision': np.zeros(len(Ks)),
        'recall': np.zeros(len(Ks)),
        'ndcg': np.zeros(len(Ks)),
        'hit_ratio': np.zeros(len(Ks)),
        'auc': 0.
    }

    print("Starting evaluation...")
    model.eval()
    with torch.no_grad():
        start_time = time()
        user_embeddings, item_embeddings = model.generate()  # Expect to count all users and items embedded
        print(f"All embeddings generated in {time() - start_time:.2f} seconds.")

        # Batch users
        user_indices = list(test_user_set.keys())
        n_batches = int(np.ceil(len(user_indices) / float(batch_size)))

        for batch_num in range(n_batches):
            batch_start_time = time()
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(user_indices))
            users_batch = user_indices[start_index:end_index]

            print(f"Processing batch {batch_num + 1}/{n_batches}... (users {start_index} to {end_index})")
            user_embedding_batch = user_embeddings[users_batch]
            scores = torch.matmul(user_embedding_batch,
                                  item_embeddings.t())  # Calculate all users' scores for all items

            for i, user in enumerate(users_batch):
                pos_items = test_user_set[user]
                all_items = set(range(n_items))
                train_items = train_user_set.get(user, [])
                test_items = list(all_items - set(train_items))

                scores_i = scores[i]
                _, indices = torch.topk(scores_i, max(Ks))
                recommends = indices.cpu().numpy()

                binary = np.isin(recommends, pos_items).astype(np.float32)
                for k_idx, K in enumerate(Ks):
                    actual_k = min(K, len(recommends))
                    relevant_items = np.isin(recommends[:actual_k], pos_items).astype(np.int)
                    result['precision'][k_idx] += precision_at_k(relevant_items, actual_k) / len(user_indices)
                    result['recall'][k_idx] += recall_at_k(relevant_items, actual_k, len(pos_items)) / len(user_indices)
                    result['ndcg'][k_idx] += ndcg_at_k(relevant_items, actual_k, pos_items) / len(user_indices)
                    result['hit_ratio'][k_idx] += hit_at_k(relevant_items, actual_k) / len(user_indices)

            print(f"Batch {batch_num + 1} processed in {time() - batch_start_time:.2f} seconds.")

    print("Evaluation complete. Total time: {:.2f} seconds".format(time() - start_time))
    return result
