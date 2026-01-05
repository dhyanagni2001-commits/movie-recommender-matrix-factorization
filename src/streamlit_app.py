import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from mf_model import MatrixFactorization


@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")

    model_path = os.path.join(artifacts_dir, "mf_model.pkl")
    maps_path = os.path.join(artifacts_dir, "id_mappings.pkl")
    movies_path = os.path.join(artifacts_dir, "movies.pkl")

    if not (os.path.exists(model_path) and os.path.exists(maps_path) and os.path.exists(movies_path)):
        raise FileNotFoundError(
            "Artifacts not found. Run train_and_recommend.py first to train and save the model."
        )

    with open(model_path, "rb") as f:
        model: MatrixFactorization = pickle.load(f)

    with open(maps_path, "rb") as f:
        id_maps = pickle.load(f)

    with open(movies_path, "rb") as f:
        movies_info = pickle.load(f)

    user2idx = id_maps["user2idx"]
    item2idx = id_maps["item2idx"]
    item_idx_to_title = movies_info["item_idx_to_title"]

    idx2user = {idx: user_id for user_id, idx in user2idx.items()}

    return {
        "model": model,
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "item_idx_to_title": item_idx_to_title,
    }


def main():
    st.title("🎬 Matrix Factorization Movie Recommender")
    st.write(
        "This app uses a matrix factorization model trained on MovieLens 100K "
        "to recommend movies to users."
    )

    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(str(e))
        st.stop()

    model: MatrixFactorization = artifacts["model"]
    idx2user = artifacts["idx2user"]
    item_idx_to_title = artifacts["item_idx_to_title"]

    user_indices = sorted(idx2user.keys())

    def user_label(u_idx: int) -> str:
        raw_id = idx2user[u_idx]
        return f"user_idx={u_idx} (raw_id={raw_id})"

    selected_user_idx = st.selectbox(
        "Select a user:",
        options=user_indices,
        format_func=user_label,
    )

    top_k = st.slider("Top-K recommendations", min_value=5, max_value=30, value=10, step=1)

    if st.button("Show recommendations"):
        rec_items, scores = model.recommend_for_user(
            user_idx=selected_user_idx,
            known_items=None,  # simple demo; could exclude known items if we pass them
            top_k=top_k,
        )

        titles = [item_idx_to_title.get(int(idx), f"item {idx}") for idx in rec_items]

        df = pd.DataFrame(
            {
                "Rank": np.arange(1, len(titles) + 1),
                "Movie": titles,
                "Item Index": rec_items,
                "Predicted Score": scores,
            }
        )

        st.subheader("Recommended Movies")
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
