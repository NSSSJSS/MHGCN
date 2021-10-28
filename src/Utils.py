import numpy as np
import torch
from scipy.io import loadmat
from scipy.sparse import csr_matrix

from src.Model import FAME_GCN


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_our_data(dataset_str, cuda=True):
    """
    Load our Networks Datasets
    Avoid awful code
    """
    data = loadmat('data/' + dataset_str + '.mat')
    # label
    try:
        labels = data['label']  # 鏌愪竴绫昏妭鐐圭殑label
    except:
        labels = data['labelmat']  # 鏌愪竴绫昏妭鐐圭殑label
    N = labels.shape[0]
    try:
        labels = labels.todense()
    except:
        pass

    # idx train valid test
    idx_train = data['train_idx'].ravel()
    try:
        idx_val = data['valid_idx'].ravel()
    except:
        idx_val = data['val_idx'].ravel()
    # idx_test = data['train_idx'].ravel()
    idx_test = data['test_idx'].ravel()
    # idx_train = np.concatenate((idx_train, idx_test))
    # node features

    try:
        node_features = data['full_feature'].toarray()
    except:
        try:
            node_features = data['feature']
        except:
            try:
                node_features = data['node_feature']
            except:
                node_features = data['features']
    features = csr_matrix(node_features)

    # edges to adj
    if dataset_str == 'small_alibaba_1_10':
        num_nodes = data['IUI_buy'].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['IUI_buy'] + data['IUI_cart'] + data["IUI_clk"] + data['IUI_collect']
    elif dataset_str == 'Aminer_10k_4class':
        num_nodes = 10000
        adj = csr_matrix((num_nodes, num_nodes))
        adj = data['PAP'] + data['PCP'] + data["PTP"]

        idx_test = idx_test - 1
        idx_train = idx_train - 1
        idx_val = idx_val - 1
    elif dataset_str == 'imdb_1_10':
        edges = data['edges'][0].tolist()
        num_nodes = edges[0].shape[0]
        adj = csr_matrix((num_nodes, num_nodes))
        for edge in edges:
            adj += edge
    else:
        num_nodes = data['A'][0][0].toarray().shape[0]
        adj = data['A'][0][0] + data['A'][0][1] + data['A'][0][2]

    print('{} node number: {}'.format(dataset_str, num_nodes))

    try:
        features = features.astype(np.int16)
    except:
        pass
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train.astype(np.int16))
    idx_val = torch.LongTensor(idx_val.astype(np.int16))
    idx_test = torch.LongTensor(idx_test.astype(np.int16))

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test


def get_model(model_opt, nfeat, nclass, A, nhid, out, dropout=0, cuda=True):

    """
     Model selection
    """

    if model_opt == "FAME_GCN":
        model = FAME_GCN(nfeat=nfeat,
                         nhid=nhid,
                         out=out,
                         dropout=dropout)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model