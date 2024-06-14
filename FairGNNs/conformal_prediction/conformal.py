import numpy as np
import torch
    
def aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha):
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        range(n), cal_labels
    ]
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(val_srt <= qhat, val_pi.argsort(axis=1), axis=1)
    
    #Set element with the largest probability True for rows where all elements are False
    #all_false_rows = ~np.any(prediction_sets, axis=1)
    #max_prob_indices = val_pi[:, 0]
    #prediction_sets[all_false_rows, max_prob_indices[all_false_rows]] = True

    cov = prediction_sets[np.arange(prediction_sets.shape[0]),val_labels].mean()
    eff = np.sum(prediction_sets)/len(prediction_sets)
    return prediction_sets, cov, eff


def pred_conformal_classification(pred, labels, calib_test_mask, calib_test_sensitive_mask, calib_test_no_sensitive_mask, n, alpha):
    # data
    #n = min(1000, int(calib_test.shape[0]/2))
    n_base = n

    logits = torch.nn.Softmax(dim = 1)(pred).detach().cpu().numpy()
    smx = logits[calib_test_mask]
    labels = labels[calib_test_mask].detach().cpu().numpy()

    cov_all = []
    eff_all = []
    sen_cov_all, nosen_cov_all = [], []
    sen_eff_all, nosen_eff_all = [], []

    #print("n base:{}".format(n_base))
    for k in range(1):
        #print("iteration within conformal classification:{}".format(k))
        idx = np.array([1] * n_base + [0] * (smx.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        cal_labels, val_labels = labels[idx], labels[~idx]


        calib_test_sen = calib_test_sensitive_mask[calib_test_mask]
        calib_test_nosen = calib_test_no_sensitive_mask[calib_test_mask]
        sen_id = calib_test_sen[~idx]
        nosen_id = calib_test_nosen[~idx]
        sen_smx, nosen_smx = val_smx[sen_id,:], val_smx[nosen_id,:]
        sen_label, nosen_label = val_labels[sen_id], val_labels[nosen_id]


        n = cal_smx.shape[0]
        # for the default setting, score == 'aps':
        prediction_sets, cov, eff = aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        sen_prediction_sets, sen_cov, sen_eff = aps(cal_smx, sen_smx, cal_labels, sen_label, n, alpha)
        nosen_prediction_sets, nosen_cov, nosen_eff = aps(cal_smx, nosen_smx, cal_labels, nosen_label, n, alpha)

        cov_all.append(cov)
        eff_all.append(eff)

        sen_cov_all.append(sen_cov)
        nosen_cov_all.append(nosen_cov)
        sen_eff_all.append(sen_eff)
        nosen_eff_all.append(nosen_eff)

    return np.mean(cov_all), np.mean(eff_all), np.mean(sen_cov_all), np.mean(sen_eff_all), np.mean(nosen_cov_all), np.mean(nosen_eff_all)

def prob_conformal_classification(pred, labels, calib_test_mask, calib_test_sensitive_mask, calib_test_no_sensitive_mask, n, alpha):
    # data
    #n = min(1000, int(calib_test.shape[0]/2))
    n_base = n

    logits = pred
    smx = logits[calib_test_mask]
    labels = labels[calib_test_mask].detach().cpu().numpy()

    cov_all = []
    eff_all = []
    sen_cov_all, nosen_cov_all = [], []
    sen_eff_all, nosen_eff_all = [], []

    #print("n base:{}".format(n_base))
    for k in range(1):
        #print("iteration within conformal classification:{}".format(k))
        idx = np.array([1] * n_base + [0] * (smx.shape[0]-n_base)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx,:], smx[~idx,:]
        cal_labels, val_labels = labels[idx], labels[~idx]


        calib_test_sen = calib_test_sensitive_mask[calib_test_mask]
        calib_test_nosen = calib_test_no_sensitive_mask[calib_test_mask]
        sen_id = calib_test_sen[~idx]
        nosen_id = calib_test_nosen[~idx]
        sen_smx, nosen_smx = val_smx[sen_id,:], val_smx[nosen_id,:]
        sen_label, nosen_label = val_labels[sen_id], val_labels[nosen_id]


        n = cal_smx.shape[0]
        # for the default setting, score == 'aps':
        prediction_sets, cov, eff = aps(cal_smx, val_smx, cal_labels, val_labels, n, alpha)
        sen_prediction_sets, sen_cov, sen_eff = aps(cal_smx, sen_smx, cal_labels, sen_label, n, alpha)
        nosen_prediction_sets, nosen_cov, nosen_eff = aps(cal_smx, nosen_smx, cal_labels, nosen_label, n, alpha)

        cov_all.append(cov)
        eff_all.append(eff)

        sen_cov_all.append(sen_cov)
        nosen_cov_all.append(nosen_cov)
        sen_eff_all.append(sen_eff)
        nosen_eff_all.append(nosen_eff)

    return np.mean(cov_all), np.mean(eff_all), np.mean(sen_cov_all), np.mean(sen_eff_all), np.mean(nosen_cov_all), np.mean(nosen_eff_all)



