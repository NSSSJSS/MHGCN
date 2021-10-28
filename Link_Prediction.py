import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import time

from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.link_prediction_evaluate import predict_model

args = get_citation_args()

# IMDB
args.dataset = 'imdb_1_10'
eval_name = r'data/imdb_1_10'
net_path = r"data/IMDB/imdb_1_10.mat"
savepath = r'data/imdb_embedding_1_10'
eval_name = r'imdb_1_10'
file_name = r'data/IMDB'
eval_type = 'all'

# DBLP
# args.dataset = 'DBLP'
# net_path = r"data/dblp/DBLP.mat"
# savepath = r'data/DBLP_embedding'
# eval_name = r'DBLP'
# file_name = r'data/dblp'
# eval_type = 'all'

# Aminer
# args.dataset = 'Aminer_10k_4class'
# eval_name = r'Aminer_10k_4class'
# net_path = r'../data/Aminer_1_13/Aminer_10k_4class.mat'
# savepath = r'embedding/Aminer_10k_4class_aminer_embedding_'
# file_name = r'../data/Aminer_1_13'
# eval_type = 'all'

# alibaba
# args.dataset = 'small_alibaba_1_10'
# eval_name = r'small_alibaba_1_10'
# net_path = r'data/small_alibaba_1_10/small_alibaba_1_10.mat'
# savepath = r'data/alibaba_embedding_'
# file_name = r'data/small_alibaba_1_10'
# eval_type = 'all'


# amazon
# args.dataset = 'amazon'
# eval_name = r'amazon'
# net_path = r'data/amazon/amazon.mat'
# savepath = r'data/amazon_embedding_'
# file_name = r'data/amazon'
# eval_type = 'all'

mat = loadmat(net_path)

try:
    train = mat['A']
except:
    try:
        train = mat['train']+mat['valid']+mat['test']
    except:
        try:
            train = mat['train_full']+mat['valid_full']+mat['test_full']
        except:
            try:
                train = mat['edges']
            except:
                train = np.vstack((mat['edge1'],mat['edge2']))

try:
    feature = mat['full_feature']
except:
    try:
        feature = mat['feature']
    except:
        try:
            feature = mat['features']
        except:
            feature = mat['node_feature']

feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

if net_path == 'imdb_1_10.mat':
    A = train[0]
elif args.dataset == 'Aminer_10k_4class':
    A = [[mat['PAP'], mat['PCP'], mat['PTP'] ]]
    feature = mat['node_feature']
    feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
else:
    A = train

node_matching = False

adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, False)
model = get_model(args.model, features.size(1), labels.max().item()+1, A, args.hidden, args.out, args.dropout, False)

starttime=time.time()
ROC, ROC, PR = predict_model(model, file_name, feature, A, eval_type, node_matching)
endtime=time.time()

print('Test ROC: {:.10f}, F1: {:.10f}, PR: {:.10f}'.format(ROC, ROC, PR))