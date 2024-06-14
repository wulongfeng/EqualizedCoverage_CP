from FairGNNs.models import CrossWalk
from FairGNNs.utils.utils import load_data

import numpy as np
from collections import defaultdict
import torch
import random

params = {
    "dataset": 'Cora',
    "num_class": 7,
    "data_seed": 0,
    "sens_attr_idx": 1352,
    "alpha": 0.1
}
# Cora, 7, 1352
# CiteSeer, 6, 2298
# PubMed, 3, 173
# CS, 15, 2862
# Physics, 5, 4403


adj, edge_index, feats, labels, idx_train, idx_val, idx_test, sens, idx_sens_train, sensitive_mask, no_sensitive_mask =load_data(params['dataset'],
                                                                                                                                 params['data_seed'],
                                                                                                                                 params['sens_attr_idx'],
                                                                                                                                 sens_number=500)
calib_test_mask = np.array([False] * len(labels))
calib_test_mask[idx_test] = True

#'''
n_sample = min(1000, int(idx_test.shape[0]/2))


# Initiate the model (with searched parameters).
model = CrossWalk()
model.fit(adj, feats, labels, idx_train, sens, params['num_class'])
# Evaluate the model.

(
    ACC,
    AUCROC,
    F1,
    cov_all,
    ineff_all,
    cov_sen,
    ineff_sen,
    cov_nosen,
    ineff_nosen
) = model.predict(idx_test, labels, calib_test_mask, sensitive_mask, no_sensitive_mask, n_sample, params['alpha'], idx_val)


for epochs in [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1500]:
    # Initiate the model (with searched parameters).
    model = CrossWalk()
    model.fit(adj, feats, labels, idx_train, sens, params['num_class'], epochs, params['dataset'])

    # Evaluate the model.
    (
        ACC,
        AUCROC,
        F1,
        cov_all,
        ineff_all,
        cov_sen,
        ineff_sen,
        cov_nosen,
        ineff_nosen
    ) = model.predict(idx_test, labels, calib_test_mask, sensitive_mask, no_sensitive_mask, n_sample, params['alpha'], idx_val)


    print("ACC:", ACC)
    print("AUCROC: ", AUCROC)
    print("F1: ", F1)
    #print("cov_all:", cov_all)
    #print("ineff_all: ", ineff_all)
    #print("cov_sen: ", cov_sen)
    #print("ineff_sen: ", ineff_sen)
    #print("cov_nosen: ", cov_nosen)
    #print("ineff_nosen: ", ineff_nosen)
    #print("ineff diff: ", abs(ineff_sen - ineff_nosen))
    print("cov_all\tineff_all\tcov_sen\tineff_sen\tcov_nosen\tineff_nosen\tcov_diff\tineff_diff")
    print("{},{},{},{},{},{},{},{}".format(cov_all, ineff_all, cov_sen, ineff_sen, cov_nosen, ineff_nosen, abs(cov_sen - cov_nosen), abs(ineff_sen - ineff_nosen)))
