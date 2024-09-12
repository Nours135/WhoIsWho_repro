import os
import sys

# use evaluation given by whoiswho
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from whoiswho.evaluation.SNDeval import evaluate

if __name__ == '__main__':
 
    predict = os.path.join(current_dir, './out/res.json')
    ground_truth = os.path.join(current_dir, './dataset/data/src/train/train_author.json')
    evaluate(predict, ground_truth)
