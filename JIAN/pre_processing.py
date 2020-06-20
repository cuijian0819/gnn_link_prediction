
from itertools import permutations, combinations


paper_author_path = "./project_data/paper_author.txt"
f1 = open(paper_author_path)
f2 = open("./project_data/processed_input.txt", "w")

f1.readline()
i = 0 
for line in f1:
    author_list = [int(author) for author in line.strip().split(" ")]

    perm_obj = permutations(author_list, 2) 
    
    for perm in perm_obj:
        f2.write(str(perm[0]) + ' ' + str(perm[1]) + '\n')


answer_public_path = "./project_data/answer_public.txt"
f_answer = open(answer_public_path)

query_public_path = "./project_data/query_public.txt"
f_query = open(query_public_path)
query = f_query.readline()
num_query = int(query.strip())
train_num = (num_query // 10) * 9

train_pos = open("./project_data/train_pos.txt", "w")
train_neg = open("./project_data/train_neg.txt", "w")
test_pos = open("./project_data/test_pos.txt", "w")
test_neg = open("./project_data/test_neg.txt", "w")

for i in range(num_query):
    query = f_query.readline()
    answer = f_answer.readline().strip()
    author_list = [int(author) for author in query.strip().split(" ")]
    comb_obj = combinations(author_list, 2) 
    for comb in comb_obj:
        if i >= train_num:
            if answer == 'True':
                test_pos.write(str(comb[0]-1) + ' ' + str(comb[1]-1) + '\n')
            else: 
                test_neg.write(str(comb[0]-1) + ' ' + str(comb[1]-1) + '\n')
        else:
            if answer == 'True':
                train_pos.write(str(comb[0]-1) + ' ' + str(comb[1]-1) + '\n')
            else: 
                train_neg.write(str(comb[0]-1) + ' ' + str(comb[1]-1) + '\n')
    
query_private_path = "./project_data/query_private.txt"
f_private = open(query_private_path)
private = f_private.readline()
num_private = int(private.strip())

test_unknown = open("./project_data/test_unknown.txt", "w")

for i in range(num_private):
    private = f_private.readline()
    author_list = [int(author) for author in private.strip().split(" ")]

    comb_obj = combinations(author_list, 2) 
    for comb in comb_obj:
        test_unknown.write(str(comb[0]-1) + ' ' + str(comb[1]-1) + '\n')

