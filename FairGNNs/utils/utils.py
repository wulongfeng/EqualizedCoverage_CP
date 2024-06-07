#%%
import numpy as np
import scipy.sparse as sp
import torch
import os
import pandas as pd
import dgl
import os.path as osp
from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

#%%
def load_data(dataset_name, data_seed, sens_attr_idx, sens_number=500):
    print('Loading {} dataset...'.format(dataset_name))

    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        pre_path = './data/Planetoid/'
        path = osp.join('data', 'Planetoid')
        dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['Physics', 'CS']:
        pre_path = './data/Coauthor/'
        path = osp.join('data', 'Coauthor')
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())

    data = dataset[0]
    #sens = pd.read_csv(pre_path + dataset_name + '/sensitive/sen_' + str(sens_attr_idx), sep = '\t', header = None)
    #sens_label = sens.iloc[:, 0].tolist()

    features = np.delete(data.x, sens_attr_idx, axis=1) #np version
    #features = torch.cat((data.x[:, :sens_attr_idx], data.x[:, sens_attr_idx + 1:]), dim=1)
    labels = data.y

    # build graph
    '''
    edges = data.edge_index
    adj = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    '''
    edges = data.edge_index
    adj_coo = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (labels.shape[0], labels.shape[0]))
    # Convert COO matrix to dense tensor (optional)
    adj_dense = adj_coo.to_dense()
    # Ensure adjacency matrix is symmetric (optional)
    adj_dense = torch.max(adj_dense, adj_dense.t())
    # Add self-loops to the adjacency matrix (optional)
    adj = adj_dense + torch.eye(adj_dense.shape[0])


    import random
    idx = np.array(range(len(data.y)))
    np.random.seed(data_seed)
    np.random.shuffle(idx)
    split_res = np.split(idx, [int(0.2 * len(idx)), int(0.3 * len(idx)), len(idx)])
    idx_train, idx_val, idx_test = split_res[0], split_res[1], split_res[2]

    print("size of train:{}, test:{}, val:{}".format(len(idx_train), len(idx_test), len(idx_val)))
    # why why why doesn't work
    #sens_label = sens.iloc[:, 0].values  # Convert to numpy array for better performance
    #sens_idx = idx[sens_label == 1]
    #print("sens_idx:{}".format(sorted(sens_idx)))
    #nosens_idx = idx[sens_label != 1]

    sens_label = np.array(data.x[:, sens_attr_idx])
    sens_label[sens_label > 0] = 1
    print("len of idx:{}, len of sens_label:{}".format(len(idx), len(sens_label)))
    sens_idx, nosens_idx = [], []
    for iidx in idx:
        if sens_label[iidx] == 1:
            sens_idx.append(iidx)
        else:
            nosens_idx.append(iidx)
    # idx_test = np.asarray(list(set(sens_idx) & set(idx_test)))
    print("size of sensitive samples:{}, non-sensitive samples:{}".format(len(sens_idx), len(nosens_idx)))
    idx_sens_train = list(set(sens_idx) - set(idx_val) - set(idx_test))
    #print("size of idx_sens_train:{}, content:{}".format(len(idx_sens_train), idx_sens_train))
    #print("size of idx_sens_test:{}, content:{}".format(len(idx_test), idx_test))
    #print("size of idx_sens_val:{}, content:{}".format(len(list(set(sens_idx) & set(idx_val))), list(set(sens_idx) & set(idx_val))))
    #print("sens_label of idx_sens_train:{}".format(sens_label[idx_sens_train]))
    #print("size of common from train with sens:{}, content:{}".format(len(list(set(sens_idx) & set(idx_train))), list(set(sens_idx) & set(idx_train))))
    random.seed(data_seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    # Create masks directly without using lists
    sensitive_mask = np.zeros(len(data.y), dtype=bool)
    sensitive_mask[sens_idx] = True
    no_sensitive_mask = np.zeros(len(data.y), dtype=bool)
    no_sensitive_mask[nosens_idx] = True

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    sens_label = torch.FloatTensor(sens_label)

    return adj, edges, features, labels, idx_train, idx_val, idx_test, sens_label, idx_sens_train, sensitive_mask, no_sensitive_mask



def load_pokec(dataset,sens_attr,predict_attr, path="../dataset/pokec/", label_number=1000,sens_number=500,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    #random.shuffle(sens_idx)
    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def feature_norm(features):

    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]

    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy(output, labels):
    output = output.squeeze()
    preds = (output>0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_softmax(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#%%



#%%
def load_pokec_emb(dataset,sens_attr,predict_attr, path="../dataset/pokec/", label_number=1000,sens_number=500,seed=19,test_idx=False):
    print('Loading {} dataset from {}'.format(dataset,path))

    graph_embedding = np.genfromtxt(
        os.path.join(path,"{}.embedding".format(dataset)),
        skip_header=1,
        dtype=float
        )
    embedding_df = pd.DataFrame(graph_embedding)
    embedding_df[0] = embedding_df[0].astype(int)
    embedding_df = embedding_df.rename(index=int, columns={0:"user_id"})

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    idx_features_labels = pd.merge(idx_features_labels,embedding_df,how="left",on="user_id")
    idx_features_labels = idx_features_labels.fillna(0)
    #%%

    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    #%%
    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]




    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train