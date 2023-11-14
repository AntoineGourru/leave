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
import itertools
import sys

def test(model, dataset, x_1_test, x_2_test, y_test, s_test):
    model.eval()

    with torch.no_grad():
        proba_p, proba_f = model(dataset.graph.x, dataset.graph.edge_index, x_1_test, x_2_test)

    return roc_auc_score(torch.squeeze(y_test), proba_p), roc_auc_score(torch.squeeze(s_test), proba_f)


if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = 'polblogs'

# Get results for different random seed
trial = 10
seeds = random.sample(range(1, 100), trial)

# ---------------- H Y P E R P A R A M E T E R S --------------------
OUTPUT_FILE = dataset_name+f"_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.txt"
HIDDEN_CHANNELS = [[256, 128]]
ALPHAS = np.arange(0, 1, 0.1)
LR = [.01]


if len(sys.argv) > 2:
    EPOCHS = [int(sys.argv[2])]
else:
    EPOCHS = [300]

if len(sys.argv) > 3:
    ENCODER = [sys.argv[3]]
else:
    ENCODER = ['GAT']

NUM_LAYERS = [2]

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

    dataset = dataset_vibgnn(train_data[0], attributes)

    dataset_train = dataset.build_train()

    criterion = torch.nn.BCELoss()

    for encoder, alpha, lr, epoch, chan, num_lay in \
            itertools.product(ENCODER, ALPHAS, LR, EPOCHS, HIDDEN_CHANNELS, NUM_LAYERS):
        print("=" * 60)
        print(f"-- Params: ENCODER={encoder} ALPHA={alpha:.2f} LR={lr:e} EPOCH={epoch:3} HIDDEN CHANNELS={chan}")
        start = time.time()

        # ==============================================================================
        # 2. Training
        # ==============================================================================

        model = VIB(num_features=dataset.graph.num_features, hidden_channels=chan, encoder=encoder, n_layers=num_lay)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
                     f"{chan[1]}],"\
                     f"NUM LAYERS={num_lay},SEED={i}"
            results = []
            for k in sorted(results_all.keys()):
                results.append(f"{k}={results_all[k]:.5f}")
            results_str = ','.join(results)
            duration = f"TIME={datetime.now()},DURATION={time.time() - start:.2f}"
            s = '\t'.join([params, results_str, duration])
            f.write(f"{s}\n")
