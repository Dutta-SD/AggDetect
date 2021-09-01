# Common utility functions

import numpy as np
import random
import string
import nltk
import re
import pandas as pd


# Random Number Seeder
def seed_all(x: int):
    random.seed(x)
    np.random.seed(x)


# Cleans one text
def clean_one_text(text: str) -> str:
    # Cleans one text and returns it

    # remove punctuation
    filter_str = string.punctuation.replace("'", "")

    new_string = text.translate(str.maketrans("", "", filter_str))
    tk = nltk.TweetTokenizer()

    s = set(nltk.corpus.stopwords.words("english"))
    # n't words
    rexp_1 = re.compile(r"n't")
    not_words = set(filter(rexp_1.findall, s))
    not_words.update(("against", "no", "nor", "not"))

    s.difference_update(not_words)

    stmr = nltk.stem.porter.PorterStemmer()
    tokens = [token for token in tk.tokenize(new_string) if token.lower() not in s]
    clean_tokens = [stmr.stem(token) for token in tokens]
    text = " ".join(clean_tokens)
    return text


def get_clean_dataset(
    df_raw, target_mapping, train=True, task_name="A", string_cleaner=None, seed=0
):
    """
    ===============================================================
    get_clean_dataset - cleans the dataset, returns text and labels
    ===============================================================

    :df_raw - pandas dataframe for cleaning
    :target_mapping - map for the targets
    :train - flag to see if training data sent or not
    :task_name - the target to predict
    :preprocessor - preprocesses the string
    :string_cleaner - useful for removing punctuation, etc(function)
    """

    # Shuffle
    df_raw = df_raw.sample(frac=1).reset_index()

    col_str = f"Sub-task {task_name}"

    if "ID" in df_raw.columns:
        df_raw = df_raw.drop(["ID"], axis=1)

    targets = df_raw[col_str].map(target_mapping).values
    text = df_raw["Text"].values.astype("str")

    if string_cleaner is not None:
        v_cleaner = np.vectorize(string_cleaner)
        text = v_cleaner(text)

    return text.reshape(-1, 1).astype("str"), targets


# Correct Training Shape
def reshape_training_data(X_train, flatten=True):
    if flatten:
        return np.ravel(X_train)
    return X_train
