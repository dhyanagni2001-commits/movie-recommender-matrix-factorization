import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils import load_ratings, load_movies, encode_ids, build_item_idx_to_title
from mf_model import MatrixFactorization
from metrics import precision_recall_at_k


# ---------- paths ----------
ratings_path = "../data/ml-100k/u.data"
items_path = "../data/ml-100k/u.item"


# ---------- load data ----------
ratings_df = load_ratings(ratings_path)
movies_df = load_movies(items_path)

# ---------- encode user & item ids ----------
ratings_df, user2idx, item2idx = encode_ids(ratings_df)

# ---------- map item index -> movie title ----------
item_idx_to_title = build_item_idx_to_title(movies_df, item2idx)

# ---------- train / test split ----------
train_df, test_df = train_test_split(
    ratings_df[["user_idx", "item_idx", "rating"]],
    test_size=0.2,
    random_state=42,
)

train_data = train_df.to_numpy()
test_data = test_df.to_numpy()

n_users = ratings_df["user_idx"].nunique()
n_items = ratings_df["item_idx"].nunique()


# ---------- create model ----------
model = MatrixFactorization(
    n_users=n_users,
    n_items=n_items,
    n_factors=40,
    lr=0.01,
    reg=0.1,
    n_epochs=20,
)

# ---------- train ----------
model.fit(train_data)


# ======================================================
# ============ RMSE / MAE EVALUATION ====================
# ======================================================

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


# ======================================================
# ========== PRECISION / RECALL @ K (ALL USERS) =========
# ======================================================

print("\n----------- Ranking Quality (Precision / Recall) -----------")

Ks = [5, 10, 20]   # multiple cutoff values

# relevant test items defined as rating >= 4
test_relevant = (
    test_df[test_df["rating"] >= 4.0]
    .groupby("user_idx")["item_idx"]
    .apply(set)
    .to_dict()
)

for K in Ks:
    all_precisions = []
    all_recalls = []

    for user_idx in ratings_df["user_idx"].unique():

        # items user already rated in training set
        known_items = set(
            train_df[train_df["user_idx"] == user_idx]["item_idx"].tolist()
        )

        # recommend top-K movies
        rec_items, _ = model.recommend_for_user(
            user_idx=user_idx,
            known_items=known_items,
            top_k=K
        )

        relevant = test_relevant.get(user_idx, set())

        p, r = precision_recall_at_k(
            recommended=list(rec_items),
            relevant=relevant,
            k=K,
        )

        all_precisions.append(p)
        all_recalls.append(r)

    print(f"\nK = {K}")
    print(f"Average Precision@{K}: {np.mean(all_precisions):.4f}")
    print(f"Average Recall@{K}:    {np.mean(all_recalls):.4f}")


# ======================================================
# =============== PRINT SAMPLE RECOMMENDATION ==========
# ======================================================

rng = np.random.default_rng(0)
user_idx = int(rng.choice(ratings_df["user_idx"].unique()))

known_items = set(train_df[train_df["user_idx"] == user_idx]["item_idx"].tolist())

rec_items, scores = model.recommend_for_user(
    user_idx=user_idx,
    known_items=known_items,
    top_k=10
)

print("\n----------- Example recommendations for a user -----------")
for rank, item_idx in enumerate(rec_items, 1):
    title = item_idx_to_title.get(int(item_idx), f"item {item_idx}")
    print(f"{rank}. {title}")
