import os
import torch.random
from vib_gnn import *
from data_loaders import DataLoader
from sklearn import preprocessing
from generate_negative import *
from sklearn.metrics import roc_auc_score
from evaluation import computeMetrics
from datetime import datetime
import scipy.sparse as sp
import itertools
import time


def test(model, dataset, x_1_test, x_2_test, y_test, s_test, A=None):
    model.eval()

    with torch.no_grad():
        if A is not None:
            proba_p, proba_f = model(dataset.graph.x, dataset.graph.edge_index, x_1_test, x_2_test, A)
        else:
            proba_p, proba_f = model(dataset.graph.x, dataset.graph.edge_index, x_1_test, x_2_test)

    return roc_auc_score(torch.squeeze(y_test), proba_p), roc_auc_score(torch.squeeze(s_test), proba_f)


# Get results for different random seed
trial = 5
seeds = random.sample(range(1, 100), trial)

# ---------------- H Y P E R P A R A M E T E R S --------------------
OUTPUT_FILE = f"polblogs_MLP_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"
ALPHAS = np.arange(0, 1, 0.1)
LR = [.01]
EPOCHS = [150]
ENCODER = ['MLP']
BETAS = [1e-9, 1e-8, 1e-7]
L = [1, 5, 10]
HIDDEN = [32, 64, 128, 256]

for i in seeds:
    seed = i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ==============================================================================
    # 1. Load and prepare data
    # ==============================================================================
    dataset_name = 'polblogs'
    data = DataLoader(dataset_name=dataset_name,
                      dataset_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    # Load and format dataset.
    data.load()

    # Get the attributes for every entity in the data.
    # Note: it is expected that these attributes are not used as features, so there is no train/test split.
    attributes = data.get_attributes()
    sens_attr_name = data.get_sens_attr_name()

    attributes = np.array(attributes)

    le = preprocessing.LabelEncoder()
    le.fit(attributes.ravel())
    attributes = le.transform(attributes.ravel())

    attributes_dic = d = dict(enumerate(attributes))

    train_data = data.get_train_data()
    test_data = data.get_test_set()

    x_1_train, x_2_train, y_train, s_train = prepare_pairs(train_data[0], attributes_dic, train_data[1])
    x_1_test, x_2_test, y_test, s_test = prepare_pairs(test_data[0], attributes_dic, test_data[1])

    dataset = dataset_vibgnn(train_data[0], attributes)
    dataset_train = dataset.build_train()

    n_nodes = torch.unique(torch.cat((x_1_train, x_2_train, x_1_test, x_2_test), 0)).size()

    A = sp.csr_matrix((np.ones(dataset_train.edge_index[0].shape),
                       (dataset_train.edge_index[0].numpy(),
                        dataset_train.edge_index[1].numpy())),
                      shape=(n_nodes[0], n_nodes[0])).toarray()
    A = torch.Tensor(A)

    criterion = torch.nn.BCELoss()

    for alpha, lr, epoch, dim, beta, l in \
            itertools.product(ALPHAS, LR, EPOCHS, HIDDEN, BETAS, L):

        print("=" * 60)
        print(f"-- Params: ENCODER=MLP ALPHA={alpha:.2f} LR={lr:e} EPOCH={epoch:3} HIDDEN DIM={dim}")
        start = time.time()

        model = VIB(num_features=dataset.graph.num_features, hidden_channels=[n_nodes[0], dim], encoder="MLP")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-6)

        for e in range(epoch):
            model.train()
            optimizer.zero_grad()  # Clear gradients.

            loss, _, _ = model.loss_vib(dataset_train.x, dataset_train.edge_index, x_1_train, x_2_train, y_train,
                                        s_train, criterion, alpha=alpha, beta=beta, L=l, A=A)

            loss.backward()
            optimizer.step()
            if e % 10 == 0:
                auc, auc_s = test(model, dataset, x_1_test, x_2_test, y_test, s_test, A)
                print(
                    f'Epoch: {e:03d}, Loss: {loss:.4f}, test AUC (Link prediction): {auc:.4f}'
                    f' test AUC for Fairness: {auc_s:.4f}')
        results_all = computeMetrics(model, dataset, x_1_test, x_2_test, y_test, s_test, A)

        with open('results/' + OUTPUT_FILE, "a") as f:  # 'a' = append mode to not overwrite
            params = f"ALPHA={alpha:.4f},LR={lr:e},EPOCH={epoch:3},ENCODER=MLP,HIDDEN CHANNELS={dim}, " \
                     f"L={l},BETA={beta},SEED = {i}"
            results = []
            for k in sorted(results_all.keys()):
                results.append(f"{k}={results_all[k]:.5f}")
            results_str = ','.join(results)
            duration = f"TIME={datetime.now()},DURATION={time.time() - start:.2f}"
            s = '\t'.join([params, results_str, duration])
            f.write(f"{s}\n")
