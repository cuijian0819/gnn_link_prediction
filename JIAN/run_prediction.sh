python Main.py --data-name paper_author --test-unknown test_unknown.txt --hop 2 --only-predict && \
echo "saving private_answer.txt to prediction..."
python result_processing.py
