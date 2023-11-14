import os
import torch.random
from vib_gnn import *
from data_loaders import DataLoader
from sklearn import preprocessing
from generate_negative import *
from sklearn.metrics import roc_auc_score
from evaluation import computeMetrics
from datetime import datetime
import time
import sys


def test(model, dataset, x_1_test, x_2_test, y_test, s_test):
    model.eval()
    with torch.no_grad():
        proba_p, proba_f = model(dataset.graph.x, dataset.graph.edge_index, x_1_test, x_2_test)

    return roc_auc_score(torch.squeeze(y_test), proba_p.cpu()), roc_auc_score(torch.squeeze(s_test), proba_f.cpu())


if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = 'polblogs'


# Get results for different random seed
trial = 1
seeds = random.sample(range(1, 100), trial)

# ---------------- H Y P E R P A R A M E T E R S --------------------
OUTPUT_FILE = dataset_name+f"_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"
chan = [256, 128]
lr = 0.01

if len(sys.argv) > 2:
    encoder = sys.argv[2]
else:
    encoder = 'GAT'

if len(sys.argv) > 3:
    alpha = float(sys.argv[3])
else:
    alpha = 0.2

num_lay = 2
epoch = 300

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for i in seeds:
    seed = i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ==============================================================================
    # 1. Load and prepare data
    # ==============================================================================


    data = DataLoader(dataset_name=dataset_name,
                      dataset_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    # Load and format dataset.
    data.load()

    # Get the attributes for every entity in the data.
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

    dataset = dataset_vibgnn(edge_list, attributes)

    edges_tohide = np.hstack([x_1_test, x_2_test])
    edges_tohide = edges_tohide[torch.squeeze(y_test.int()).numpy() == 1].astype(int)
    edges_tohide =list(zip(edges_tohide[:,0], edges_tohide[:,1]))
    dataset_train = dataset.build_train(edges_tohide)

    '''

    criterion = torch.nn.BCELoss()

    print("=" * 60)
    print(f"-- Params: ENCODER={encoder} ALPHA={alpha:.2f} LR={lr:e} EPOCH={epoch:3} HIDDEN CHANNELS={chan}")
    start = time.time()

    # ==============================================================================
    # 2. Training
    # ==============================================================================

    model = VIB(num_features=dataset.graph.num_features, hidden_channels=chan, encoder=encoder, n_layers=num_lay)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset.graph.x, dataset.graph.edge_index, x_1_test, x_2_test = dataset.graph.x.to(
        device), dataset.graph.edge_index.to(device), x_1_test.to(device), x_2_test.to(device)
    dataset_train.x, dataset_train.edge_index, x_1_train, x_2_train, y_train, s_train = dataset_train.x.to(
        device), dataset_train.edge_index.to(device), x_1_train.to(device), x_2_train.to(device), y_train.to(
        device), s_train.to(device)

    for e in range(epoch):
        model.train()
        optimizer.zero_grad()  # Clear gradients.

        # loss_vib : vib based loss, loss: soft cont loss without stochasticity
        loss, _, _ = model.loss(dataset_train.x, dataset_train.edge_index, x_1_train, x_2_train, y_train,
                                s_train, criterion, alpha)
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            auc_y, auc_s = test(model, dataset, x_1_test, x_2_test, y_test, s_test)
            print(
                f'Epoch: {e:03d}, Loss: {loss:.4f}, test AUC (Link prediction): {auc_y:.4f}'
                f', test AUC (Sensitive group): {auc_s:.4f}')

    # ==============================================================================
    # 3. Evaluation
    # ==============================================================================
    results_all = computeMetrics(model, dataset, x_1_test, x_2_test, y_test, s_test)

    # ==========================================================================
    # 4. Save results
    # ==========================================================================
    with open('results/'+dataset_name+'/'+OUTPUT_FILE, "a") as f:  # 'a' = append mode to not overwrite
        params = f"ALPHA={alpha:.4f},LR={lr:e},EPOCH={epoch:3}, ENCODER={encoder}, HIDDEN CHANNELS=[{chan[0]}-" \
                 f"{chan[1]}]," \
                 f"NUM LAYERS={num_lay},SEED={i}"
        results = []
        for k in sorted(results_all.keys()):
            results.append(f"{k}={results_all[k]:.5f}")
        results_str = ','.join(results)
        duration = f"TIME={datetime.now()},DURATION={time.time() - start:.2f}"
        s = '\t'.join([params, results_str, duration])
        f.write(f"{s}\n")
        '''
