from itertools import permutations, combinations


f_pred_answer = open('./prediction/pred_tf.txt')
f_pred_score = open('./prediction/pred_score.txt')
f_pred_link = open('./prediction/pred_link.txt')

private_public_path = "./project_data/query_private.txt"
f_w1 = open('./prediction/prediction_tf.txt', 'w')
f_w2 = open('./prediction/prediction_score.txt', 'w')

f_private = open(private_public_path)
private = f_private.readline()
num_private = int(private.strip())

pred_link2answer = {}
pred_link2score = {}

for i in range(num_private):
    pred_link = tuple(f_pred_link.readline().strip().split(" "))
    print(f_pred_answer)
    pred_answer = int(f_pred_answer.readline().strip())
    pred_score = float(f_pred_score.readline().strip())
    pred_link2answer[pred_link] = pred_answer
    pred_link2score[pred_link] = pred_score

for i in range(num_private):
    private = f_private.readline()
    author_tp = tuple([author for author in private.strip().split(" ")])
    if author_tp in pred_link2answer and author_tp in pred_link2score:
        f_w1.write(str(pred_link2answer[author_tp]) + '\n')
        f_w2.write(str(pred_link2score[author_tp]) + '\n')
    else:
        print("should not happen")
        print(author_tp)
        print(pred_link2answer[0])
        break
