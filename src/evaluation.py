from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import numpy as np
import torch


def deltaEO(y_true, y_pred, sens_attr):
    opp_ex = [x for x in zip(y_true, y_pred, sens_attr) if x[2] == 1]
    same_ex = [x for x in zip(y_true, y_pred, sens_attr) if x[2] == 0]

    opp_yt, opp_yp, _ = zip(*opp_ex)
    same_yt, same_yp, _ = zip(*same_ex)

    opp_tpr = conf_matrix(opp_yt, opp_yp)
    same_tpr = conf_matrix(same_yt, same_yp)

    delta_EO_val = abs(same_tpr - opp_tpr)

    return delta_EO_val


def disparate_impact(y_pred, sens_attr):
    index = []
    c = 0
    same_group_count, opp_group_count = 0, 0
    for ind in range(len(y_pred)):
        if sens_attr[ind] == 0:
            same_group_count += y_pred[ind]
        else:
            opp_group_count += y_pred[ind]

    return opp_group_count / same_group_count


def conf_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    TPR = tp / (tp + fn)
    return TPR


def rep_bias(emb, sens_attr, onevsrest=True):
    if onevsrest:
        classes = list(set(sens_attr))
        y = label_binarize(sens_attr, classes=classes)
        model = OneVsRestClassifier(LogisticRegression(max_iter=5000))
        params = {'estimator__C': 100. ** np.arange(-1, 1), }
        clf = GridSearchCV(model, params, cv=2, scoring='roc_auc')
        clf.fit(emb, y)
        test_score = clf.best_score_
        return test_score
    else:
        kfold = KFold(n_splits=10, random_state=55, shuffle=True)
        model_kfold = LogisticRegression(solver='lbfgs', max_iter=300)
        results_kfold = cross_val_score(model_kfold, emb, sens_attr, cv=kfold, scoring='roc_auc')
        auc_orig_mean = results_kfold.mean()
        return auc_orig_mean


def computeMetrics(model, dataset, x_1_test, x_2_test, y_test, s_test, A=None):
    with torch.no_grad():
        
        model.eval()
        
        if A is not None:
            proba_p, _ = model(dataset.graph.x, dataset.graph.edge_index, x_1_test, x_2_test, A)
        else:
            proba_p, _ = model(dataset.graph.x, dataset.graph.edge_index, x_1_test, x_2_test)
            
        y_pred = torch.squeeze(proba_p).cpu().numpy()
        y_pred_argmax = np.round(y_pred)

        auc_link = roc_auc_score(y_test, y_pred)
        print(f"AUC {auc_link:.4f}")

        delta = deltaEO(torch.squeeze(y_test).numpy(), y_pred_argmax, s_test)
        print(f"Delta EO {delta:.4f}")

        di = disparate_impact(y_pred, s_test)
        print(f"DI {delta:.4f}")
        
        if A is not None:
            embedding = model.encoder.fc1(model.encoder.do(A))
            embedding = model.encoder.tanh(embedding)
            embedding = model.encoder.fc2(model.encoder.do(embedding))
        else:
            embedding = model.gnn(dataset.graph.x, dataset.graph.edge_index)

        repbias = rep_bias(embedding.cpu().numpy(), dataset.graph.y.cpu().numpy())
        print(f"Rep Bias {repbias:.4f}")

    return {'AUC': auc_link, 'RB': repbias, 'EO': delta, 'DI': di}
