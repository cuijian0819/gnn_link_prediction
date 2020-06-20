from itertools import permutations, combinations


f = open('./prediction/private_pred.txt')

f_w = open('./prediction/private_answer.txt', 'w')

private_public_path = "./project_data/query_private.txt"
f_private = open(private_public_path)
private = f_private.readline()
num_private = int(private.strip())

for i in range(num_private):
    private = f_private.readline()
    author_list = [int(author) for author in private.strip().split(" ")]

    comb_obj = combinations(author_list, 2) 
    
    answer = 'True\n'

    for j in range(len(list(comb_obj))): 
        line = f.readline()
        (n1, n2, score) = tuple(line.strip().split(' '))
        if float(score) < 0.3:
            answer = 'False\n'
            break;

    f_w.write(answer)

