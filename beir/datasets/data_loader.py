from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv

logger = logging.getLogger(__name__)

class GenericDataLoader:
    
    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl", query_file: str = "queries.jsonl", 
                 qrels_folder: str = "qrels", qrels_file: str = "", candidate_lm: list = ['human'], target: str = 'human', temperature: list = [0.2]):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        # assert target in candidate_lm
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_files = []
        for source in candidate_lm:
            if source == "human":
                self.corpus_files.append(os.path.join(data_folder, 'corpus-human.jsonl'))
            else:
                for t in temperature:
                    self.corpus_files.append(os.path.join(data_folder, os.path.join(str(t), 'corpus-{}.jsonl'.format(source))))
        logger.info("Corpus Files {}".format(self.corpus_files))
        self.data_folder = data_folder
        self.candidate_lm = candidate_lm
        self.target = target
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))
        
        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        for corpus_file in self.corpus_files:
            self.check(fIn=corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
        
        # if not len(self.corpus):
        for corpus_file in self.corpus_files:
            logger.info("Loading Corpus...")
            self._load_corpus(corpus_file)
        logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
        logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels} # remove queries that are not in qrels
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus
    
    def _load_corpus(self, corpus_file):
        
        num_lines = sum(1 for i in open(corpus_file, 'rb'))
        generate_type = corpus_file.split('corpus-')[1].split('.jsonl')[0]
        if generate_type == 'human':
            generate_type = 'human'
        else:
            target_t = corpus_file.split('/')[-2]
            generate_type = generate_type + '_{}'.format(target_t)
        with open(corpus_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                if generate_type == self.target:
                    self.corpus[line.get("_id")] = {
                        "text": line.get("text"),
                        "title": "",
                    }
                else:
                    self.corpus[line.get("_id")+"-"+generate_type] = {
                        "text": line.get("text"),
                        "title": "",
                    }
    
    def _load_queries(self):
        
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = line.get("text")
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score
