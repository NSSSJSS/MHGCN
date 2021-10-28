import torch
from scipy.io import loadmat
from scipy.sparse import csc_matrix

from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.node_classfication_evaluate import node_classification_evaluate

args = get_citation_args()

# IMDB
args.dataset = 'imdb_1_10'
eval_name = r'data/imdb_1_10'
net_path = r"data/IMDB/imdb_1_10.mat"
savepath = r'data/imdb_embedding_1_10'
eval_name = r'imdb_1_10'

# DBLP
# args.dataset = 'DBLP'
# eval_name = r'data/DBLP'
# net_path = r"data/dblp/DBLP.mat"
# savepath = r'data/DBLP_embedding'
# eval_name = r'DBLP'

# Aminer
# args.dataset = 'Aminer_10k_4class'
# eval_name = r'Aminer_10k_4class'
# net_path = r'../data/Aminer_1_13/Aminer_10k_4class.mat'
# savepath = r'embedding/Aminer_10k_4class_aminer_embedding_'
# eval_name = r'Aminer_10k_4class'

# alibaba
# args.dataset = 'small_alibaba_1_10'
# eval_name = r'small_alibaba_1_10'
# net_path = r'data/small_alibaba_1_10/small_alibaba_1_10.mat'
# savepath = r'data/alibaba_embedding_'
# eval_name = r'small_alibaba_1_10'

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
            train = mat['edges']

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
A = train[0][0]
feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

if net_path == 'imdb_1_10.mat':
    A = train[0]
elif args.dataset == 'Aminer_10k_4class':
    A = [[mat['PAP'], mat['PCP'], mat['PTP'] ]]
    feature = mat['node_feature']
    feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
else:
    A = train


adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, args.cuda)

model = get_model(args.model, features.size(1), labels.max().item()+1, A, args.hidden, args.out, args.dropout, False)

f1_ma, f1_mi = node_classification_evaluate(model, feature, A, eval_name, file_type='mat', device=torch.device('cpu'))

print('Test F1-ma: {:.10f}, F1-mi: {:.10f}'.format(f1_ma, f1_mi))