import sys, os
sys.path.append(os.path.dirname((os.path.dirname(os.path.abspath(__file__)))))

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import TFIDFSearch as TFIDF

import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "--test_dataset", type=str, choices=["scifact"], help="The dataset test on", default="scifact")
parser.add_argument(
        "--target", type=str, choices=["human", "llama-2-7b-chat", "gpt-3.5-turbo-0613"], help="The target model test on", default="human")
parser.add_argument(
        "--version", type=int, choices=[0,1,2,3], default=2)
parser.add_argument(
        '--candidate_lm', nargs='+', type=str, help="The candidate test model list", default=["human"])
parser.add_argument(
        '--temperature', nargs='+', type=float, default=[0.2])
parser.add_argument(
        '--target_t', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = args.test_dataset

if args.target != 'human':
    target = args.target + "_{}".format(args.target_t)
else:
	target = args.target

candidate_lm = args.candidate_lm
temperature = args.temperature
data_base_path = "./datasets/"
logging.info("The test dataset is {}".format(dataset))
logging.info("The test target model is {}".format(target))
logging.info("The candidate_lm is {}".format(candidate_lm))
data_path = data_base_path + "{}".format(args.test_dataset)

corpus, queries, qrels = GenericDataLoader(data_path, candidate_lm=candidate_lm, target=target, temperature=temperature).load(split="test")

model = TFIDF(version=2)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
