from pygdebias.debiasing import EDITS
from pygdebias.datasets import Bail
from pygdebias.debiasing.utils import load_data

import numpy as np
from collections import defaultdict
import torch
import random

'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
setup_seed(11)

bail = Bail()
#bail = Cora()
adj, feats, idx_train, idx_val, idx_test, labels, sens = (
    bail.adj(),
    bail.features(),
    bail.idx_train(),
    bail.idx_val(),
    bail.idx_test(),
    bail.labels(),
    bail.sens(),
)

'''
params = {
    "dataset": 'Cora',
    "num_class": 7,
    "data_seed": 0,
    "sens_attr_idx": 30,
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
#adj_data = adj_csr.data
#adj_indices = torch.from_numpy(np.vstack((adj_data.row, adj_data.col)))
# Create a sparse PyTorch tensor
#adj = torch.sparse_coo_tensor(adj_indices, adj_data, adj_data.shape)
#'''


# Initiate the model (with searched parameters).
model = EDITS(feats).cuda()

model.fit(adj, feats, sens, idx_train, idx_val)

# Evaluate the model.
(
    ACC,
    AUCROC,
    F1,
    ACC_sens0,
    AUCROC_sens0,
    F1_sens0,
    ACC_sens1,
    AUCROC_sens1,
    F1_sens1,
    SP,
    EO,
) = model.predict(adj, labels, sens, idx_train, idx_val, idx_test)

print("ACC:", ACC)
print("AUCROC: ", AUCROC)
print("F1: ", F1)
print("ACC_sens0:", ACC_sens0)
print("AUCROC_sens0: ", AUCROC_sens0)
print("F1_sens0: ", F1_sens0)
print("ACC_sens1: ", ACC_sens1)
print("AUCROC_sens1: ", AUCROC_sens1)
print("F1_sens1: ", F1_sens1)
print("SP: ", SP)
print("EO:", EO)
