'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''

from sentence_transformers import models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever, Debiased_MultipleNegativesRankingLoss
import pathlib, os
import logging
import argparse
import random
import copy
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
                    type=str,
                    help="The dataset test on",
                    default="msmarco")
parser.add_argument("--LLM",
                    type=str,
                    help="Training data of LLM-generated",
                    default="llama-2-7b-chat")
parser.add_argument("--num_epochs",
                    type=int,
                    help="The number of epochs",
                    default=10)
parser.add_argument("--batch_size",
                    type=int,
                    help="The batch size",
                    default=64)
parser.add_argument("--score_func",
                    type=str,
                    help="The score function",
                    default="dot",
                    choices=["dot", "cos"])
parser.add_argument("--alpha",
                    type=float,
                    help="The alpha for constraint_loss",
                    default=0.1)
parser.add_argument("--model_name",
                    type=str,
                    help="The model name",
                    default="distilbert-base-uncased")
parser.add_argument("--output_dir",
                    type=str,
                    help="The output path",)
args = parser.parse_args()

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset
LLM = args.LLM
num_epochs = args.num_epochs
batch_size = args.batch_size
alpha = args.alpha
model_name = args.model_name
output_dir = args.output_dir
logging.info("The test dataset is {}".format(dataset))
logging.info("The LLM-generated training data is from {}".format(LLM))
logging.info("The num_epochs is {}".format(num_epochs))
logging.info("The batch_size is {}".format(batch_size))
logging.info("The alpha is {}".format(alpha))
logging.info("The model_name is {}".format(model_name))
logging.info("The output_dir is {}".format(output_dir))
logging.info("The score function is {}".format(args.score_func))


# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

data_path = f"{dataset}"

#### Provide the data_path where nfcorpus has been downloaded and unzipped
# corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
human_corpus, queries, qrels = GenericDataLoader(data_folder=data_path, corpus_file="corpus.jsonl").load(split="train")
llm_corpus, queries, qrels = GenericDataLoader(data_folder=data_path, corpus_file=f"corpus-{LLM}.jsonl").load(split="train")

# directly finetune the whole sentence-transformer model
if model_name == "distilbert-base-uncased":
    word_embedding_model = models.Transformer(model_name, max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#### Or provide pretrained sentence-transformer model
else:
    model = SentenceTransformer(model_name)

retriever = TrainRetriever(model=model, batch_size=batch_size)

#### Prepare training samples
train_samples = retriever.load_train(human_corpus, llm_corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Training SBERT with cosine-product defualt similarity_fct = util.cos_sim
if args.score_func == "cos":
    train_loss = Debiased_MultipleNegativesRankingLoss(
        model=retriever.model, similarity_fct=util.cos_sim, alpha=alpha)
elif args.score_func == "dot":
    train_loss = Debiased_MultipleNegativesRankingLoss(
        model=retriever.model, similarity_fct=util.dot_score, alpha=alpha, scale=1)

# #### Prepare dev evaluator
# ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
# ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(output_dir, f"{model_name}-{LLM}-{dataset}-{args.score_func}-alpha{alpha}")
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
# evaluate after each epoch
# evaluation_steps = int(len(train_samples) / retriever.batch_size) + 1
evaluation_steps = 0
# warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
warmup_steps = 1000

retriever.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=None,
              epochs=num_epochs,
              output_path=model_save_path,
              warmup_steps=warmup_steps,
              evaluation_steps=evaluation_steps,
              use_amp=True)
