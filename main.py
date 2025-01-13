import argparse
import json
import pickle
import random
import warnings
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.linear_model import LogisticRegression
from knn import get_top_k_closest_and_furthest, cosine_similarity_torch, get_top_k_similar_and_dissimilar, \
    get_sample_sets
from loss import compute_loss
from model import Model
from utils import load_network_data, random_planetoid_splits, get_train_data
from knn import normalize_features


warnings.filterwarnings('ignore')

def train():
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    adj1, features, Y = load_network_data(args.dataset)
    features[features > 0] = 1

    n = features.shape[0]
    d = features.shape[1]
    c = Y.shape[1]
    features = torch.FloatTensor(features.todense()).to(device)  

    labels = np.argmax(Y, 1)
    adj = torch.tensor(adj1.todense()).to(device)  
    adj = torch.nonzero(adj, as_tuple=False).t()

    adjs = []
    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj, num_nodes=n)
    adjs.append(adj)

    e = adj.shape[1]
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

    model = Model(d, args.hidden, args.output).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)
    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda(device)

    cnt_wait = 0
    best_loss = 1e9
    best_epoch = 0

    distance_matrix = np.load(args.filenpy)
    closest_nodes, furthest_nodes = get_top_k_closest_and_furthest(distance_matrix, args.k)

    for epoch in range(args.nb_epochs):
        model.train()
        model.zero_grad()

        # Forward pass for GCN
        x_gcn,output_nodeformer = model(features, adj,adjs)
        x_nodeformer , l = output_nodeformer
        l = torch.tensor(l, device=args.device)
        Y_indices = np.argmax(Y, axis=1)
        # Compute cosine similarities
        gcn_similarity = cosine_similarity_torch(x_gcn)
        nodeformer_similarity = cosine_similarity_torch(x_nodeformer)

        # Find similar and dissimilar nodes
        gcn_most_similar = get_top_k_similar_and_dissimilar(gcn_similarity, args.k)
        nodeformer_most_similar = get_top_k_similar_and_dissimilar(nodeformer_similarity,
                                                                                            args.k)
        del gcn_similarity
        del nodeformer_similarity
        torch.cuda.empty_cache()

        positive_samples = []
        gcn_sum = 0
        nodeformer_sum = 0
        common_similar_set_sum = 0
        knn_sum = 0
        positive_set_sum = 0
        real_po_sum = 0
        for node in range(n):
            gcn_similar_set = gcn_most_similar[node]
            nodeformer_similar_set = nodeformer_most_similar[node]
            common_similar_set = np.intersect1d(gcn_similar_set, nodeformer_similar_set)
            positive_set, negative_set = get_sample_sets(n, common_similar_set, closest_nodes[node])
            positive_samples.append(positive_set)

        loss = compute_loss(n, x_gcn, x_nodeformer, positive_samples, device=device,tau=0.5,
                            batch_size=args.batch_size)
        print(f'Epoch: {epoch}, Loss: {loss}')
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            cnt_wait = 0
            torch.save({
                'model': model.state_dict(),
            }, args.best_model_path)
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        loss.backward()  
        model_optimizer.step()
    print(f'Loading best model from epoch {best_epoch}')

    checkpoint = torch.load(args.best_model_path)
    model.load_state_dict(checkpoint['model'],False)

    model.eval()

    embeds_gcn = model.model_gcn(features, adj)
    embeds_gcn = normalize_features(embeds_gcn)
    output_nodeformer = model.model_nodeformer(features, adjs)
    embeds_nodeformer, _ = output_nodeformer
    embeds_nodeformer = normalize_features(embeds_nodeformer)
    embeds = args.alpha * embeds_gcn + (1 - args.alpha) * embeds_nodeformer
    with open("node_embeddings.pkl", 'wb') as f:
        pickle.dump(embeds, f)  
    print("Node embeddings saved to node_embeddings.pkl")
    embeds = embeds.detach().cpu()


    Accuaracy_test_allK = []
    numRandom = 20

    for train_num in [20]:

        AccuaracyAll = []
        for random_state in range(numRandom):
            print(
                "\n=============================%d-th random split with training num %d============================="
                % (random_state + 1, train_num))

            if train_num == 20:
                if args.dataset in ['cora']:
                    val_num = 500
                    idx_train, idx_val, idx_test = random_planetoid_splits(c, torch.tensor(labels), train_num,
                                                                           random_state)
                else:
                    val_num = 30
                    idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

            else:
                val_num = 0  
                idx_train, idx_val, idx_test = get_train_data(Y, train_num, val_num, random_state)

            train_embs = embeds[idx_train, :]
            val_embs = embeds[idx_val, :]
            test_embs = embeds[idx_test, :]

            if train_num == 20:
                best_val_score = 0.0
                for param in [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]:
                    LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, C=param)
                    LR.fit(train_embs, labels[idx_train])
                    val_score = LR.score(val_embs, labels[idx_val])
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_parameters = {'C': param}

                LR_best = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0, **best_parameters)

                LR_best.fit(train_embs, labels[idx_train])
                y_pred_test = LR_best.predict(test_embs)  # pred label
                print("Best accuracy on validation set:{:.4f}".format(best_val_score))
                print("Best parameters:{}".format(best_parameters))

            else:  
                LR = LogisticRegression(solver='liblinear', multi_class='ovr', random_state=0)
                LR.fit(train_embs, labels[idx_train])
                y_pred_test = LR.predict(test_embs)  # pred label

            test_acc = accuracy_score(labels[idx_test], y_pred_test)
            print("test accuaray:{:.4f}".format(test_acc))
            AccuaracyAll.append(test_acc)

        average_acc = np.mean(AccuaracyAll) * 100
        std_acc = np.std(AccuaracyAll) * 100
        print('avg accuracy over %d random splits: %.1f +/- %.1f, for train_num: %d, val_num:%d\n' % (
            numRandom, average_acc, std_acc, train_num, val_num))
        Accuaracy_test_allK.append(average_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,help="random seed")
    parser.add_argument("--filenpy", type=str, default='distance_matrix_cora.npy', help="topology matrix")
    parser.add_argument("--nb_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--device", type=int, default=0, help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--dataset", type=str, default="cora", help="which dataset for training")
    parser.add_argument("--k", type=int, default=520, help="Number of top similar and dissimilar nodes")
    parser.add_argument("--hidden", type=int, default=512, help="layer hidden")
    parser.add_argument("--output", type=int, default=440, help="layer output")
    parser.add_argument("--alpha", type=float, default=0.7, help="output bili")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--l2_coef", type=float, default=5e-5, help="L2 regularization coefficient")
    parser.add_argument("--batch_size", type=int, default=300, help="Batch size")
    parser.add_argument("--sparse", type=bool, default=False, help="Sparse or not")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--best_model_path", type=str, default='best_model.pkl', help="Path to save the best model")
    train()
