import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from torch.utils.data import DataLoader
sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *


parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--data-name', default=None, help='network name', required=True)
parser.add_argument('--train-pos', default=None, help='train pos')
parser.add_argument('--train-neg', default=None, help='train neg')
parser.add_argument('--test-pos', default=None, help='test pos')
parser.add_argument('--test-neg', default=None, help='test neg')
parser.add_argument('--test-unknown', default=None, help='test unknown')

parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predictions\
                    for links in test-name; you still need to specify train-name\
                    in order to build the observed network and extract subgraphs')

parser.add_argument('--epoch', default=50, help='epoch', required=False)

parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,...')
parser.add_argument('--max-nodes-per-hop', default=200, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='save the final model')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
max_query_node = 0

args.file_dir = os.path.dirname(os.path.realpath('__file__'))

# check whether train and test links are provided
train_pos, train_neg, test_pos, test_neg = None, None, None, None
if args.train_pos is not None:
    args.train_dir = os.path.join(args.file_dir, './project_data/{}'.format(args.train_pos))

    f = open(args.train_dir)
    len2train_idx = {}
    len2train_pos = {}
    for line in f:
        author_list = [int(author) for author in line.strip().split(" ")]
        if len(author_list) > max_query_node:
            max_query_node = len(author_list)

        num_author = len(author_list)
        if num_author not in len2train_idx:
            len2train_idx[num_author] = np.zeros((0,num_author))
        len2train_idx[num_author] = np.append(len2train_idx[num_author], [author_list], axis=0)
    f.close()

    # print(len2train_idx)
    for length in len2train_idx:
        len2train_pos[length] = tuple() 
        for i in range(length):
            len2train_pos[length] += (len2train_idx[length][:, i], )

    # print(len2train_pos)

if args.train_neg is not None:
    args.train_dir = os.path.join(args.file_dir, './project_data/{}'.format(args.train_neg))

    f = open(args.train_dir)
    len2train_idx = {}
    len2train_neg = {}
    for line in f:
        author_list = [int(author) for author in line.strip().split(" ")]
        if len(author_list) > max_query_node:
            max_query_node = len(author_list)

        num_author = len(author_list)
        if num_author not in len2train_idx:
            len2train_idx[num_author] = np.zeros((0,num_author))
        len2train_idx[num_author] = np.append(len2train_idx[num_author], [author_list], axis=0)
    f.close()

    for length in len2train_idx:
        len2train_neg[length] = tuple() 
        for i in range(length):
            len2train_neg[length] += (len2train_idx[length][:, i], )


if args.test_pos is not None:
    args.test_dir = os.path.join(args.file_dir, './project_data/{}'.format(args.test_pos))
    f = open(args.test_dir)
    len2test_idx = {}
    len2test_pos = {}
    for line in f:
        author_list = [int(author) for author in line.strip().split(" ")]
        if len(author_list) > max_query_node:
            max_query_node = len(author_list)

        num_author = len(author_list)
        if num_author not in len2test_idx:
            len2test_idx[num_author] = np.zeros((0,num_author))
        len2test_idx[num_author] = np.append(len2test_idx[num_author], [author_list], axis=0)
    f.close()

    for length in len2test_idx:
        len2test_pos[length] = tuple() 
        for i in range(length):
            len2test_pos[length] += (len2test_idx[length][:, i], )


if args.test_neg is not None:
    args.test_dir = os.path.join(args.file_dir, './project_data/{}'.format(args.test_neg))
    f = open(args.test_dir)
    len2test_idx = {}
    len2test_neg = {}
    for line in f:
        author_list = [int(author) for author in line.strip().split(" ")]
        if len(author_list) > max_query_node:
            max_query_node = len(author_list)

        num_author = len(author_list)
        if num_author not in len2test_idx:
            len2test_idx[num_author] = np.zeros((0,num_author))
        len2test_idx[num_author] = np.append(len2test_idx[num_author], [author_list], axis=0)
    f.close()

    for length in len2test_idx:
        len2test_neg[length] = tuple() 
        for i in range(length):
            len2test_neg[length] += (len2test_idx[length][:, i], )

if args.test_unknown is not None:
    args.test_dir = os.path.join(args.file_dir, './project_data/{}'.format(args.test_unknown))
    f = open(args.test_dir)

    len2test_idx = {}
    len2test_unknown = {}
    for line in f:
        author_list = [int(author) for author in line.strip().split(" ")]
        if len(author_list) > max_query_node:
            max_query_node = len(author_list)

        num_author = len(author_list)
        if num_author not in len2test_idx:
            len2test_idx[num_author] = np.zeros((0,num_author))
        len2test_idx[num_author] = np.append(len2test_idx[num_author], [author_list], axis=0)
    f.close()

    for length in len2test_idx:
        len2test_unknown[length] = tuple() 
        for i in range(length):
            len2test_unknown[length] += (len2test_idx[length][:, i], )

# build observed network
if args.data_name is not None:  # use .mat network
    args.data_dir = os.path.join(args.file_dir, './project_data/{}.mat'.format(args.data_name))
    data = sio.loadmat(args.data_dir)
    net = data['net']
    if 'group' in data:
        # load node attributes (here a.k.a. node classes)
        attributes = data['group'].toarray().astype('float32')
    else:
        attributes = None
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
else:
    print("SHOULD NOT HAPPEN")
    exit(-1)

# ground-truth max label
n = max_query_node
d = (5 * (n**2) - n) / 2 
max_label = 3 + (d//2) * ( (d//2) + (d%2) -1 )

if max_label > 10000:
    max_label = 10000

print("max number of query node: ", max_query_node)
print("ground-truth max label: ", max_label)

'''Train and apply classifier'''
A = net.copy()  # the observed network
A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

node_information = None
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    node_information = embeddings

if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

if args.only_predict:  # no need to use negatives
    _, test_graphs, max_n_label = links2subgraphs(
        A, 
        None, 
        None, 
        len2test_unknown, 
        None, 
        args.hop, 
        args.max_nodes_per_hop, 
        node_information, 
        args.no_parallel
    )
    print('# test: %d' % (len(test_graphs)))
else:
    train_graphs, test_graphs, max_n_label = links2subgraphs(
        A, 
        len2train_pos, 
        len2train_neg, 
        len2test_pos, 
        len2test_neg, 
        args.hop, 
        args.max_nodes_per_hop,
        node_information,
        args.no_parallel
    )
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

assert(max_n_label <= max_label)
print("max label in subgraph: ", max_n_label)

# DGCNN configurations
if args.only_predict:
    print("Model is loading...")
    with open('model/{}_hyper.pkl'.format(args.data_name), 'rb') as hyperparameters_name:
        saved_cmd_args = pickle.load(hyperparameters_name)
    
    for key, value in vars(saved_cmd_args).items(): # replace with saved cmd_args
        vars(cmd_args)[key] = value
    
    classifier = Classifier(predict = True)
    
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    model_name = 'model/{}_model.pth'.format(args.data_name)

    print("prediction is based on: {}".format(model_name))

    classifier.load_state_dict(torch.load(model_name))
    classifier.eval()
    
    predictions_tf = []
    predictions_score = []
    batch_graph = []
    pred_link_list = []
    for i, graph in enumerate(test_graphs):
        batch_graph.append(graph)
        pred_link_list.append(graph.pred_link)
        if len(batch_graph) == cmd_args.batch_size or i == (len(test_graphs)-1):
            #print(classifier(batch_graph)[:,0].tolist())
            (score, answer) = classifier(batch_graph)
            predictions_tf += answer[:,0].tolist()
            predictions_score.append(score[:, 1].exp().cpu().detach())
            batch_graph = []
    predictions_score = torch.cat(predictions_score, 0).unsqueeze(1).numpy()

    pred_tf_name = 'prediction/' + 'pred_tf.txt'
    pred_score_name = 'prediction/' + 'pred_score.txt'
    pred_link_name = 'prediction/' + 'pred_link.txt'
    np.savetxt(pred_tf_name, predictions_tf, fmt=['%d'])
    np.savetxt(pred_score_name, predictions_score, fmt=['%.3f'])
    f = open(pred_link_name ,'w')
    for pred_link in pred_link_list:
        line = ' '.join(str(int(link)+1) for link in pred_link)
        f.write(line + '\n')
    
    print('Predictions for are saved! \n')
    exit()


cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.num_epochs = int(args.epoch)
cmd_args.learning_rate = 1e-4
cmd_args.printAUC = True
cmd_args.feat_dim = int(max_label + 1)
cmd_args.attr_dim = 0


if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
    cmd_args.sortpooling_k = max(10, num_nodes_list[k_])
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

random.shuffle(train_graphs)
val_num = int(0.1 * len(train_graphs))
val_graphs = train_graphs[:val_num]
train_graphs = train_graphs[val_num:]

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(
        train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=args.batch_size
    )
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

    classifier.eval()
    val_loss = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))))
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
    print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        epoch, val_loss[0], val_loss[1], val_loss[2]))
    if best_loss is None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        best_epoch = epoch
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))))
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2]))

print('\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
    best_epoch, test_loss[0], test_loss[1], test_loss[2]))
        
if args.save_model:
    model_name = 'model/{}_model.pth'.format(args.data_name)
    print('Saving final model states to {}...'.format(model_name))
    torch.save(classifier.state_dict(), model_name)
    hyper_name = 'model/{}_hyper.pkl'.format(args.data_name)
    with open(hyper_name, 'wb') as hyperparameters_file:
        pickle.dump(cmd_args, hyperparameters_file)
        print('Saving hyperparameters to {}...'.format(hyper_name))

with open('acc_results.txt', 'a+') as f:
    f.write(str(test_loss[1]) + '\n')

if cmd_args.printAUC:
    with open('auc_results.txt', 'a+') as f:
        f.write(str(test_loss[2]) + '\n')

