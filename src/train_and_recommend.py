import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_utils import load_ratings, load_movies, encode_ids, build_item_idx_to_title
from mf_model import MatrixFactorization
from metrics import precision_recall_at_k, ndcg_at_k


# ----------------- paths -----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_DIR = os.path.join(BASE_DIR, "data", "ml-100k")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

ratings_path = os.path.join(DATA_DIR, "u.data")
items_path = os.path.join(DATA_DIR, "u.item")


# ----------------- load data -----------------
ratings_df = load_ratings(ratings_path)
movies_df = load_movies(items_path)

# encode IDs -> indices
ratings_df, user2idx, item2idx = encode_ids(ratings_df)

# item index -> title mapping
item_idx_to_title = build_item_idx_to_title(movies_df, item2idx)

# train / test split
train_df, test_df = train_test_split(
    ratings_df[["user_idx", "item_idx", "rating"]],
    test_size=0.2,
    random_state=42,
)

train_data = train_df.to_numpy()
test_data = test_df.to_numpy()

n_users = ratings_df["user_idx"].nunique()
n_items = ratings_df["item_idx"].nunique()

print(f"Users: {n_users}, Items: {n_items}, Ratings: {len(ratings_df)}")


# ----------------- model -----------------
model = MatrixFactorization(
    n_users=n_users,
    n_items=n_items,
    n_factors=40,
    lr=0.01,
    reg=0.1,
    n_epochs=20,
)


# ----------------- training -----------------
model.fit(train_data)


# ----------------- rating prediction quality (RMSE / MAE) -----------------
def compute_rmse_mae(model, test_data):
    user_ids = test_data[:, 0].astype(int)
    item_ids = test_data[:, 1].astype(int)
    ratings = test_data[:, 2].astype(float)

    preds = []
    for u, i in zip(user_ids, item_ids):
        scores = model.predict_for_user(u)
        preds.append(scores[i])

    preds = np.array(preds)
    errors = preds - ratings

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    return rmse, mae


rmse, mae = compute_rmse_mae(model, test_data)
print("\n----------- Rating Prediction Quality -----------")
print(f"Test RMSE = {rmse:.4f}")
print(f"Test MAE  = {mae:.4f}")


# ----------------- ranking quality (Precision / Recall / NDCG) -----------------
print("\n----------- Ranking Quality (Precision / Recall / NDCG) -----------")

Ks = [5, 10, 20]

# relevant test items: rating >= 4
test_relevant = (
    test_df[test_df["rating"] >= 4.0]
    .groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

all_precisions_k10 = None
all_recalls_k10 = None
all_ndcgs_k10 = None

for K in Ks:
    all_precisions = []
    all_recalls = []
    all_ndcgs = []

    for user_idx in ratings_df["user_idx"].unique():
        # items seen in training by this user
        known_items = set(
            train_df[train_df["user_idx"] == user_idx]["item_idx"].tolist()
        )

        # recommend top-K movies
        rec_items, _ = model.recommend_for_user(
            user_idx=user_idx,
            known_items=known_items,
            top_k=K,
        )

        relevant = test_relevant.get(user_idx, set())

        p, r = precision_recall_at_k(
            recommended=list(rec_items),
            relevant=relevant,
            k=K,
        )
        n = ndcg_at_k(
            recommended=list(rec_items),
            relevant=relevant,
            k=K,
        )

        all_precisions.append(p)
        all_recalls.append(r)
        all_ndcgs.append(n)

    avg_p = float(np.mean(all_precisions))
    avg_r = float(np.mean(all_recalls))
    avg_n = float(np.mean(all_ndcgs))

    print(f"\nK = {K}")
    print(f"Average Precision@{K}: {avg_p:.4f}")
    print(f"Average Recall@{K}:    {avg_r:.4f}")
    print(f"Average NDCG@{K}:      {avg_n:.4f}")

    if K == 10:
        all_precisions_k10 = all_precisions
        all_recalls_k10 = all_recalls
        all_ndcgs_k10 = all_ndcgs


# ----------------- per-user histograms (for K=10) -----------------
if all_precisions_k10 is not None:
    print("\nSaving histograms for per-user Precision/Recall/NDCG@10 in artifacts/ ...")

    # Precision@10 histogram
    plt.figure()
    plt.hist(all_precisions_k10, bins=20)
    plt.xlabel("Precision@10")
    plt.ylabel("Number of users")
    plt.title("Per-user Precision@10")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "precision_at_10_hist.png"))
    plt.close()

    # Recall@10 histogram
    plt.figure()
    plt.hist(all_recalls_k10, bins=20)
    plt.xlabel("Recall@10")
    plt.ylabel("Number of users")
    plt.title("Per-user Recall@10")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "recall_at_10_hist.png"))
    plt.close()

    # NDCG@10 histogram
    plt.figure()
    plt.hist(all_ndcgs_k10, bins=20)
    plt.xlabel("NDCG@10")
    plt.ylabel("Number of users")
    plt.title("Per-user NDCG@10")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "ndcg_at_10_hist.png"))
    plt.close()


# ----------------- save model + mappings for Streamlit -----------------
print("\nSaving model and mappings to artifacts/ ...")

model_path = os.path.join(ARTIFACTS_DIR, "mf_model.pkl")
maps_path = os.path.join(ARTIFACTS_DIR, "id_mappings.pkl")
movies_path = os.path.join(ARTIFACTS_DIR, "movies.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(maps_path, "wb") as f:
    pickle.dump(
        {
            "user2idx": user2idx,
            "item2idx": item2idx,
        },
        f,
    )

with open(movies_path, "wb") as f:
    pickle.dump(
        {
            "item_idx_to_title": item_idx_to_title,
        },
        f,
    )

print(f"Saved model to {model_path}")
print(f"Saved ID mappings to {maps_path}")
print(f"Saved movie titles to {movies_path}")


# ----------------- sample recommendations for one user -----------------
rng = np.random.default_rng(0)
user_idx_sample = int(rng.choice(ratings_df["user_idx"].unique()))

known_items_sample = set(
    train_df[train_df["user_idx"] == user_idx_sample]["item_idx"].tolist()
)

rec_items_sample, scores_sample = model.recommend_for_user(
    user_idx=user_idx_sample,
    known_items=known_items_sample,
    top_k=10,
)

print("\n----------- Example recommendations for a random user -----------")
print(f"user_idx = {user_idx_sample}")
for rank, (item_idx, score) in enumerate(zip(rec_items_sample, scores_sample), 1):
    title = item_idx_to_title.get(int(item_idx), f"item {item_idx}")
    print(f"{rank}. {title}  (score={score:.3f})")
