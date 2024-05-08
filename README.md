# LLM4IR-Bias

The official repository of the paper "LLMs may Dominate Information Access: Neural Retrievers are Biased Towards LLM-Generated Texts".  [[arXiv](http://arxiv.org/abs/2310.20501)] 

&#x1F4A1; We will update more results and datasets in a few weeks!


## Citation
```
If you find our code or work useful for your research, please cite our work.

@article{dai2023llms,
  title={Llms may dominate information access: Neural retrievers are biased towards llm-generated texts},
  author={Dai, Sunhao and Zhou, Yuqi and Pang, Liang and Liu, Weihao and Hu, Xiaolin and Liu, Yong and Zhang, Xiao and Xu, Jun},
  journal={arXiv preprint arXiv:2310.20501},
  year={2023}
}
```

## Quick Start

- For details of datasets, please check file `datasets/README.md`

- For details of evaluating codes, please check the code in the folder `evaluate/`

- For details of dataloader code, please check the file `beir/datasets/data_loader.py`

## File Structure

```shell
.
├── beir  # * evaluating codes from beir
│   ├── datasets # * codes for datalaoder
│   ├── reranking # * codes for reranking model
│   └── retrieval # * codes for lexical and dense retrieval model 
├── datasets
│   ├── 0.2 # * corpus generted by LLM with temperature 0.2
│   ├── 1.0 # * corpus generted by LLM with temperature 1.0
│   └── qrels # * relevance for queries
└── evaluate  # * codes for evaluating different retrieval model
```

## Quick Start Example with Contriever

```bash
# test on human corpus
python evaluate/evaluate_contriever.py --test_dataset scifact \
    --target human --candidate_lm human

# test on llama-2-7b-chat corpus
python evaluate/evaluate_contriever.py --test_dataset scifact \
    --target llama-2-7b-chat --candidate_lm llama-2-7b-chat

# test metric targeting on human-written on mix-corpora
python evaluate/evaluate_contriever.py --test_dataset scifact \
    --target human --candidate_lm human llama-2-7b-chat

# test metric targeting on LLM-generated on mix-corpora
python evaluate/evaluate_contriever.py --test_dataset scifact \
    --target llama-2-7b-chat --candidate_lm human llama-2-7b-chat
```

## Dependencies

The Cocktail benchmark is built based on [BEIR](https://github.com/beir-cellar/beir) and [Sentence Transformers](https://huggingface.co/sentence-transformers).

This repository has the following dependency requirements.

```
python==3.10.13
pandas==2.1.4
scikit-learn==1.3.2
evaluate==0.4.1
sentence-transformers==2.2.2
spacy==3.7.2
tiktoken==0.5.2
pytrec-eval==0.5
```

The required packages can be installed via `pip install -r requirements.txt`.
