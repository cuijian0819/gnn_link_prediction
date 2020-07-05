python Main.py \
--data-name paper_author \
--train-pos train_pos.txt \
--train-neg train_neg.txt \
--test-pos test_pos.txt \
--test-neg test_neg.txt \
--hop 2 \
--max-nodes-per-hop 100 \
--save-model \
--epoch 100 \
&& \

echo "Finish Training"
