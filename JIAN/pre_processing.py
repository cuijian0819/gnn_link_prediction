from itertools import permutations, combinations


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
test_known = open("./project_data/test_known.txt", "w")

for i in range(num_query):
    query = f_query.readline()
    answer = f_answer.readline().strip()
    author_list = [int(author)-1 for author in query.strip().split(" ")]
    query_1 = ' '.join(str(author) for author in author_list)
    test_known.write(query_1 + '\n')
    
    if i >= train_num:
        if answer == 'True':
            test_pos.write(query_1 + '\n')
        else: 
            test_neg.write(query_1 + '\n')
    else:
        if answer == 'True':
            train_pos.write(query_1 + '\n')
        else: 
            train_neg.write(query_1 + '\n')


query_private_path = "./project_data/query_private.txt"
f_private = open(query_private_path)
private = f_private.readline()
num_private = int(private.strip())

test_unknown = open("./project_data/test_unknown.txt", "w")

for i in range(num_private):
    private = f_private.readline()
    author_list = [int(author)-1 for author in private.strip().split(" ")]
    query_1 = ' '.join(str(author) for author in author_list)
    test_unknown.write(query_1 + '\n')
