from sentence_transformers import SentenceTransformer, SentencesDataset, datasets
from sentence_transformers.evaluation import SentenceEvaluator, SequentialEvaluator, InformationRetrievalEvaluator
from sentence_transformers.readers import InputExample
from transformers import AdamW
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.autonotebook import trange
from typing import Dict, List, Callable, Iterable, Tuple
import logging
import time
import random
from .. import util

logger = logging.getLogger(__name__)

class TrainRetriever:
    
    def __init__(self, model: SentenceTransformer, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size

    def load_train(self, human_corpus: Dict[str, Dict[str, str]], llm_corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> List[InputExample]:
        
        query_ids = list(queries.keys())
        train_samples = []

        for idx, start_idx in enumerate(trange(0, len(query_ids), self.batch_size, desc='Adding Input Examples')):
            query_ids_batch = query_ids[start_idx:start_idx+self.batch_size] # 每次取batch个query
            for query_id in query_ids_batch:
                for corpus_id, score in qrels[query_id].items():
                    if score >= 1: # if score = 0, we don't consider for training
                        try:
                            query = queries[query_id]
                            human_doc = human_corpus[corpus_id].get("text")
                            llm_doc = llm_corpus[corpus_id].get("text")
                            train_samples.append(InputExample(guid=idx, texts=[query, human_doc, llm_doc], label=1))
                        except KeyError:
                            logging.error("Error: Key {} not present in corpus!".format(corpus_id))

        logger.info("Loaded {} training pairs.".format(len(train_samples)))
        return train_samples


    def prepare_train(self, train_dataset: List[InputExample], shuffle: bool = True, dataset_present: bool = False) -> DataLoader:
        
        if not dataset_present: 
            train_dataset = SentencesDataset(train_dataset, model=self.model)
        
        train_dataloader = DataLoader(train_dataset, shuffle=shuffle, batch_size=self.batch_size)
        return train_dataloader

    
    def load_ir_evaluator(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], 
                 qrels: Dict[str, Dict[str, int]], max_corpus_size: int = None, name: str = "eval") -> SentenceEvaluator:

        if len(queries) <= 0:
            raise ValueError("Dev Set Empty!, Cannot evaluate on Dev set.")
        
        rel_docs = {}
        corpus_ids = set()
        
        # need to convert corpus to cid => doc      
        corpus = {idx: corpus[idx].get("title") + " " + corpus[idx].get("text") for idx in corpus}
        
        # need to convert dev_qrels to qid => Set[cid]        
        for query_id, metadata in qrels.items():
            rel_docs[query_id] = set()
            for corpus_id, score in metadata.items():
                if score >= 1:
                    corpus_ids.add(corpus_id)
                    rel_docs[query_id].add(corpus_id)
        
        if max_corpus_size:
            # check if length of corpus_ids > max_corpus_size
            if len(corpus_ids) > max_corpus_size:
                raise ValueError("Your maximum corpus size should atleast contain {} corpus ids".format(len(corpus_ids)))
            
            # Add mandatory corpus documents
            new_corpus = {idx: corpus[idx] for idx in corpus_ids}
            
            # Remove mandatory corpus documents from original corpus
            for corpus_id in corpus_ids:
                corpus.pop(corpus_id, None)
            
            # Sample randomly remaining corpus documents
            for corpus_id in random.sample(list(corpus), max_corpus_size - len(corpus_ids)):
                new_corpus[corpus_id] = corpus[corpus_id]

            corpus = new_corpus

        logger.info("{} set contains {} documents and {} queries".format(name, len(corpus), len(queries)))
        return InformationRetrievalEvaluator(queries, corpus, rel_docs, name=name)


    def fit(self, 
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Optimizer = AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            **kwargs):
        
        # Train the model
        logger.info("Starting to Train...")

        self.model.fit(train_objectives=train_objectives,
                evaluator=evaluator,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                warmup_steps=warmup_steps,
                optimizer_class=optimizer_class,
                scheduler=scheduler,
                optimizer_params=optimizer_params,
                weight_decay=weight_decay,
                output_path=output_path,
                evaluation_steps=evaluation_steps,
                save_best_model=save_best_model,
                max_grad_norm=max_grad_norm,
                use_amp=use_amp,
                callback=callback, **kwargs)


class Debiased_MultipleNegativesRankingLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.

        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.

        The performance usually increases with increasing batch sizes.

        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)

        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.

    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, alpha: float = 0.1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(Debiased_MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.margin_rank_loss = nn.MarginRankingLoss()


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        query_rep, human_doc_rep, llm_doc_rep = reps
        
        human_scores = self.similarity_fct(query_rep, human_doc_rep) * self.scale
        llm_scores = self.similarity_fct(query_rep, llm_doc_rep) * self.scale
        human_pos_scores = torch.diag(human_scores).reshape(-1, 1)
        llm_pos_scores = torch.diag(llm_scores).reshape(-1, 1)

        mask = torch.ones_like(human_scores) - torch.eye(human_scores.shape[0], device=human_scores.device)
        human_neg_scores = human_scores[mask.bool()].reshape(-1, human_scores.shape[1]-1)
        llm_neg_scores = llm_scores[mask.bool()].reshape(-1, llm_scores.shape[1]-1)
        scores_1 = torch.cat([human_pos_scores, human_neg_scores, llm_neg_scores], dim=1)
        scores_2 = torch.cat([llm_pos_scores, human_neg_scores, llm_neg_scores], dim=1)
        labels_1 = torch.zeros(len(scores_1), dtype=torch.long, device=scores_1.device)
        labels_2 = torch.zeros(len(scores_2), dtype=torch.long, device=scores_1.device)
        loss = self.cross_entropy_loss(scores_1, labels_1) + self.cross_entropy_loss(scores_2, labels_2)
        
        constraint_loss = self.margin_rank_loss(human_pos_scores, llm_pos_scores, torch.ones_like(human_pos_scores))
        
        return loss + self.alpha * constraint_loss

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__}
