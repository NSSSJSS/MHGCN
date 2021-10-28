import numpy as np
import torch
from scipy.sparse import coo_matrix


def coototensor(A):
    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def adj_matrix_weight_merge(A, adj_weight): # Multiplex Relation Aggregation

    N = A[0][0].shape[0]
    temp = coo_matrix((N, N))
    temp = coototensor(temp)


    # Alibaba
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[1][0].tocoo())
    # c = coototensor(A[2][0].tocoo())
    # d = coototensor(A[3][0].tocoo())
    # A_t = torch.stack([a, b, c, d], dim=2).to_dense()

    # DBLP
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # c = coototensor(A[0][2].tocoo())
    # A_t = torch.stack([a, b, c], dim=2).to_dense()

    # Aminer
    # a = coototensor(A[0][0].tocoo())
    # b = coototensor(A[0][1].tocoo())
    # c = coototensor(A[0][2].tocoo())
    # A_t = torch.stack([a, c], dim=2).to_dense()


    # IMDB
    a = coototensor(A[0][0].tocoo())
    b = coototensor(A[0][2].tocoo())
    A_t = torch.stack([a, b], dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)
    # temp = temp.to_sparse()

    return temp + temp.transpose(0, 1)