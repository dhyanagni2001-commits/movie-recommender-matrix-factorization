import pandas as pd
from typing import Dict, Tuple


def load_ratings(path: str) -> pd.DataFrame:
    col_names = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(path, sep="\t", names=col_names, engine="python")
    return df


def load_movies(path: str):
    """
    Load MovieLens u.item file, which is pipe-separated and has many columns.
    We only read: movie_id and title.
    """

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],                  # only first two columns
        names=["movie_id", "title"],     # assign names to them
        engine="python",
    )
    return df


def encode_ids(ratings_df: pd.DataFrame):
    user_ids = ratings_df["user_id"].unique()
    item_ids = ratings_df["item_id"].unique()

    user2idx: Dict[int, int] = {u: idx for idx, u in enumerate(user_ids)}
    item2idx: Dict[int, int] = {m: idx for idx, m in enumerate(item_ids)}

    df = ratings_df.copy()
    df["user_idx"] = df["user_id"].map(user2idx)
    df["item_idx"] = df["item_id"].map(item2idx)

    return df, user2idx, item2idx


def build_item_idx_to_title(movies_df, item2idx):
    item_idx_to_title = {}
    for movie_id, title in zip(movies_df["movie_id"], movies_df["title"]):
        if movie_id in item2idx:
            item_idx = item2idx[movie_id]
            item_idx_to_title[item_idx] = str(title)
    return item_idx_to_title
