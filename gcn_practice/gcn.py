import torch
import torch.nn as nn
import torch.nn.functional as F

import networkx as nx
from itertools import permutations, combinations

import time
from datetime import datetime

def normalize(A , symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # Degree of all nodes
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)


class GCN(nn.Module):

    def __init__(self , A, dim_in , dim_out):
        super(GCN, self).__init__()
        self.A = A
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_in//2, bias=False)
        self.fc3 = nn.Linear(dim_in//2, dim_out, bias=False)

    def forward(self,X):
        # 3 layer
        X = F.relu(self.fc1(self.A.mm(X)))
        X = F.relu(self.fc2(self.A.mm(X)))
        return self.fc3(self.A.mm(X))

# Load Graph G

'''
paper_author_path = "./project_data/paper_author.txt"
    
G = nx.Graph()

f = open(paper_author_path)

f.readline()

for line in f:
    author_list = [int(author) for author in line.strip().split(" ")]
  
    comb = combinations(author_list, 2) 
    #print(list(perm))
    G.add_edges_from(list(comb))
'''
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G).todense()

# A_normed = D^(-1/2)AD^(-1/2)
A_normed = normalize(torch.FloatTensor(A),True)

N = len(A)

X_dim = N

X = torch.eye(N,X_dim)

Y = torch.zeros(N).long()

Y_mask = torch.zeros(N, 1, dtype=torch.int)

'''
query_public_path = "./project_data/query_public.txt"
f_query = open(query_public_path)

query = f_query.readline()
num_query = int(query.strip())

for i in range(num_query):
    query = f_query.readline()
    answer = f_answer.readline().strip()
    author_list = [int(author) for author in query.strip().split(" ")]

    perm = permutations(author_list, 2) 
'''

# One sample for each classification
Y[0]=0
Y[N-1]=1

# Mask for extracting sample 
Y_mask[0][0]=1
Y_mask[N-1][0]=1


# Real Data
'''
f_answer = open(answer_public_path)
answer_public_path = "./project_data/answer_public.txt"
'''

Real = torch.zeros(N , dtype=torch.long)
for i in [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22] :
    Real[i-1] = 0
for i in [9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34] :
    Real[i-1] = 1

# Graph convolutional neural network model
gcn = GCN(A_normed ,X_dim, 2)
print(gcn(X))
# Adam Optimizer
gd = torch.optim.Adam(gcn.parameters())

'''
print(gcn.parameters())
for param in gcn.parameters():
    print(param)
'''

for i in range(300):
    #print(datetime.now())

    y_pred =F.softmax(gcn(X),dim=1)
    #print(y_pred)
    
    # cross entropy
    loss = nn.CrossEntropyLoss()

    output = loss(y_pred, Real)
    #print(Real)
    '''
    loss = (-y_pred.log().gather(1,Y))
    loss = loss.masked_select(Y_mask).mean()
    '''

    # clear previous gradient 
    gd.zero_grad()

    # calculate loss
    output.backward()

    # update 
    gd.step()

    if i%20==0 :
        # xwwprint(y_pred)
        # print(y_pred.max(1))
        _,mi = y_pred.max(1)
        print(mi)
        # 计算精确度
        print((mi == Real).float().mean())




