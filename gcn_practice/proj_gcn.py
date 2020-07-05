import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
from itertools import permutations, combinations

from datetime import datetime

def normalize(A , symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # Degree of all nodes
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d, -1))
        return D.mm(A)

class GCN(nn.Module):
    def __init__(self, A, dim_in, dim_out):
        super(GCN, self).__init__()
        self.A = A
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in//2, bias=False)
        self.fc3 = nn.Linear(dim_in//2, dim_out, bias=False)
        self.M = torch.nn.Parameter(torch.eye(len(A), len(A)))
        self.M.requires_grad = True

    def forward(self, X1, X2):
        # 3 layer
        X1 = F.relu(self.fc1(self.A.mm(X1)))
        X1 = F.relu(self.fc2(self.A.mm(X1)))
        X1 = F.relu(self.fc3(self.A.mm(X1)))

        X2 = F.relu(self.fc1(self.A.mm(X2)))
        X2 = F.relu(self.fc2(self.A.mm(X2)))
        X2 = F.relu(self.fc3(self.A.mm(X2)))

        X1_T = X1.t()

        return X1_T.mm(self.M).mm(X2)

Y_list = []
query_list = []

G = nx.Graph()

paper_author_path = "./project_data/paper_author.txt"
f = open(paper_author_path)

f.readline()
i = 0 
for line in f:
    author_list = [int(author) for author in line.strip().split(" ")]
    edge_list = []
    if len(author_list) == 1:
        G.add_node(author_list[0])
        continue;
    elif len(author_list) == 2:
        comb = combinations(author_list, 2) 
        edge_list = list(comb)
    else:
        comb = combinations(author_list, 2) 
        edge_list = list(comb)
    #print(edge_list)
    G.add_nodes_from(author_list)
    G.add_edges_from(edge_list)
    i += 1
    if i == 1000:
        break


A = nx.adjacency_matrix(G).todense()
print("finish densifiying graph: ", datetime.now())

# A_normed = D^(-1/2)AD^(-1/2)
A_normed = normalize(torch.FloatTensor(A),True)

print("finish normalizing A: ", datetime.now())

answer_public_path = "./project_data/answer_public.txt"
f_answer = open(answer_public_path)

query_public_path = "./project_data/query_public.txt"
f_query = open(query_public_path)
query = f_query.readline()
num_query = int(query.strip())

for i in range(num_query):
    query = f_query.readline()
    answer = f_answer.readline().strip()
    if answer == True: 
        answer = 1
    else: 
        answer = 0

    author_list = [int(author) for author in query.strip().split(" ")]

    perm_list = permutations(author_list, 2)
    for perm in perm_list:
        query_list.append(perm)
        Y_list.append(int(answer))

    if i == 5:
        break

Y = torch.FloatTensor(Y_list)
print("finish calculating Y: ", datetime.now())

X1 = torch.eye(len(A), len(A))
X2 = torch.eye(len(A), len(A))

# Graph convolutional neural network model
gcn = GCN(A_normed ,len(A), 1)

# Adam Optimizer
gd = torch.optim.Adam(gcn.parameters())

for i in range(300):

    y_pred = gcn(X1, X2).data

    loss = nn.CrossEntropyLoss()

    output = loss(y_pred, Real)

    gd.zero_grad()

    # calculate loss
    output.backward()

    # update 
    gd.step()



















