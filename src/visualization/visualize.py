import pandas as pd
import json
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# load data into pandas
with open("./datasets/recipes.jl", "r", encoding="utf-8") as f:
    lines = f.read().splitlines()
line_dicts = [json.loads(line) for line in lines]
for line_dict in line_dicts:
    del line_dict["body"]
recipes = pd.DataFrame(line_dicts)
recipe_names = recipes["recipe_name"].values

# visualize word counts with histogram
cutoff = 10  # discard n-grams with less than x occurrences
stop_words = ["39"]
unigram_vectorizer = CountVectorizer(
    strip_accents="unicode", ngram_range=(1, 1), min_df=1e-4
)
unigrams = unigram_vectorizer.fit_transform(recipe_names)

bigram_vectorizer = CountVectorizer(
    strip_accents="unicode", ngram_range=(3, 3), min_df=cutoff, stop_words=stop_words
)
bigrams = bigram_vectorizer.fit_transform(recipe_names)

bigram_df = (
    pd.DataFrame(
        data=bigrams.toarray(), columns=bigram_vectorizer.get_feature_names_out()
    )
    .sum(axis=0)
    .sort_values(ascending=False)
)
bigram_df.head(20).plot(kind="barh")
plt.show()
pass
# map relative embedding locations using sklearn tsne

# data cleaning???
