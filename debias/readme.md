Here, we provide an example of implementation of our debiasing method equipped with [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html?highlight=loss#sentence_transformers.losses.MultipleNegativesRankingLoss).

1. Replace the original beir/retrieval/train.py as train.py in this folder.
2. Run debias.py.