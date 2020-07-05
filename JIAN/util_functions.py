from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../pytorch_DGCNN' % cur_dir)
sys.path.append('%s/software/node2vec/src' % cur_dir)
from util import GNNGraph
import multiprocessing as mp
from itertools import islice
from itertools import permutations, combinations
import math
    
def links2subgraphs(A, len2train_pos, len2train_neg, len2test_pos, len2test_neg, h=1, 
                    max_nodes_per_hop=None, node_information=None, no_parallel=False):
    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links, g_label):
        g_list = []
        length = len(links)
        if no_parallel:
            zip_args = [links[length][k] for k in range(length)]
            for tp in tqdm(zip(*zip_args)):
              tp = tuple(map(int, tp))
              g, n_labels, n_features, ind = subgraph_extraction_labeling(
                tp, A, h, max_nodes_per_hop, node_information
              )
              max_n_label['value'] = max(max(n_labels), max_n_label['value'])
              g_list.append(GNNGraph(g, g_label, n_labels, n_features, tp))
            
            return g_list
        
        else:
            # the parallel extraction code
            start = time.time()
            pool = mp.Pool(mp.cpu_count())

            tp_list = []
            for length in links:
                zip_args = [links[length][k] for k in range(length)]
                tp_list += [tuple(map(int, tp)) for tp in zip(*zip_args)]
            results = pool.map_async(
                parallel_worker, 
                [(tp, A, h, max_nodes_per_hop, node_information) for tp in tp_list]
            )

            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            g_list = [GNNGraph(g, g_label, n_labels, ind, n_features) for g, n_labels, n_features, ind in results]
            max_n_label['value'] = max(
                max([max(n_labels) for _, n_labels, _, _ in results]), max_n_label['value']
            )
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs, test_graphs = None, None
    if len2train_pos and len2train_neg:
        train_graphs = helper(A, len2train_pos, 1) + helper(A, len2train_neg, 0)
    if len2test_pos and len2test_neg:
        test_graphs = helper(A, len2test_pos, 1) + helper(A, len2test_neg, 0)
    elif len2test_pos:
        test_graphs = helper(A, len2test_pos, 1)
    return train_graphs, test_graphs, max_n_label['value']

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None,
                                 node_information=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    nodes = set([ind[i] for i in range(len(ind))])
    visited = set([ind[i] for i in range(len(ind))])
    fringe = set([ind[i] for i in range(len(ind))])
    nodes_dist = [0, 0]
    for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
    # move target nodes to top

    for i in range(len(ind)):
        nodes.remove(ind[i])

    nodes = [ind[i] for i in range(len(ind))] + list(nodes) 
    
    subgraph = A[nodes, :][:, nodes]

    # apply node-labeling
    labels = node_label(subgraph, ind, h)

    # get node features
    features = None
    if node_information is not None:
        features = node_information[nodes]

    # construct nx graph
    g = nx.from_scipy_sparse_matrix(subgraph)
    
    # remove link between target nodes
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)

    # features = None
    return g, labels.tolist(), features, ind


def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res


def node_label(subgraph, ind, h):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]

    dist_to_node = []
    for i in range(len(ind)):
      dist_to_node_i = ssp.csgraph.shortest_path(subgraph, directed=False, unweighted=True)[i][len(ind):]
      dist_to_node.append(dist_to_node_i)

    d = sum(dist_to_node).astype(int)

    d_over_2, d_mod_2 = np.divmod(d, 2)

    while (len(dist_to_node) > 2):
        x = dist_to_node.pop()        
        y = dist_to_node.pop()     
        min_dist = np.minimum(x, y).astype(int)
        dist_to_node.append(list(min_dist))

    min_dist = np.minimum(*dist_to_node).astype(int)

    labels = 1 + min_dist + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1 for i in range(len(ind))]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    
    assert(K == len(labels))

    return labels
    
def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False, train_neg=None):
    if negative_injection:
        row, col = train_neg
        A = A.copy()
        A[row, col] = 1  # inject negative train
        A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=16, walk_length=200)
    walks = list(list(map(str, walk)) for walk in walks)
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, 
            workers=8, iter=16)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings


def CalcAUC(sim, test_pos, test_neg):
    pos_scores = np.asarray(sim[test_pos[0], test_pos[1]]).squeeze()
    neg_scores = np.asarray(sim[test_neg[0], test_neg[1]]).squeeze()
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

