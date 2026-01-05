🎬 ##**Movie Recommendation System using Matrix Factorization**

A complete end-to-end recommender system built from scratch using Matrix Factorization (MF) on the MovieLens-100K dataset.

**This project implements:**

✔️ Collaborative Filtering with latent factor models

✔️ Explicit feedback training (ratings 1–5)

✔️ Bias-aware MF (user + item bias + global mean)

✔️ Evaluation metrics:

RMSE / MAE

Precision@K

Recall@K

NDCG@K

✔️ Per-user metric histograms

✔️ Interactive Streamlit UI for browsing recommendations

✔️ Saved model artifacts for reuse

This project is built without using surprise/lightfm/recommenders library — core MF is implemented manually for learning clarity.

**🚀 Demo (What the project does)**

trains matrix factorization on MovieLens 100K

learns hidden user and movie representations

recommends top-K movies to any user

evaluates recommendation quality

visualizes how well model performs per user

launches an interface where you can:

select a user

choose K

view recommended movies

**📂 Project Structure**
movie-recommender-matrix-factorization
│
├── data/
│   └── ml-100k/                # MovieLens dataset
│
├── src/
│   ├── data_utils.py           # data loading & preprocessing
│   ├── mf_model.py             # matrix factorization model
│   ├── metrics.py              # evaluation metrics
│   ├── train_and_recommend.py  # training + evaluation pipeline
│   └── streamlit_app.py        # UI for recommendations
│
├── artifacts/
│   ├── mf_model.pkl            # trained model
│   ├── id_mappings.pkl         # user/item index mapping
│   ├── movies.pkl              # movie title mapping
│   ├── precision_at_10_hist.png
│   ├── recall_at_10_hist.png
│   └── ndcg_at_10_hist.png
│
└── README.md

**🧠 What is Matrix Factorization?**

User–item ratings matrix is mostly empty.

We approximate it as:

R ≈ P × Qᵀ


Where:

R → user-item rating matrix

P → user latent vectors (n_users × k)

Q → item latent vectors (n_items × k)

Prediction formula:

r̂_ui = μ + b_u + b_i + p_u · q_i


Where:

μ = global average rating

bᵤ = user bias

bᵢ = item bias

pᵤ, qᵢ = k-dim latent vectors

We train parameters by minimizing:

MSE + L2 regularization


using Stochastic Gradient Descent.

**📊 Results (your actual model’s performance)**

Your trained model achieved:

RMSE = ~0.94
MAE  = ~0.74


Ranking metrics:

Metric	K=5	K=10	K=20
Precision@K	~0.03	~0.046	~0.048
Recall@K	~0.01	~0.028	~0.067
NDCG@K	~0.028	~0.042	~0.055

Interpretation:

RMSE/MAE are solid for MovieLens-100K

recall increases with K (expected)

NDCG consistent with ranking quality

baseline MF — no tuning yet

**🛠 Installation & Setup**

1️⃣ Clone repo
git clone <your-repo-url>
cd movie-recommender-matrix-factorization

2️⃣ (Recommended) create virtual environment
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt


(If you didn’t make requirements.txt yet, install manually:)

numpy
pandas
scikit-learn
matplotlib
streamlit

**📥 Download dataset**

Download MovieLens-100K:

https://grouplens.org/datasets/movielens/100k/

Unzip and place into:

data/ml-100k/


So that files like u.data exist there.

**🧾 Train the model**

From inside src:

cd src
python3 train_and_recommend.py


This will:

✔ train MF
✔ compute metrics
✔ save model artifacts
✔ generate histograms

Artifacts are saved in:

artifacts/

**📈 Visualizations**

Generated automatically:

precision_at_10_hist.png

recall_at_10_hist.png

ndcg_at_10_hist.png

These show distribution across users, not just averages.

**🖥 Run Streamlit UI**

From project root:

streamlit run src/streamlit_app.py


You will see:

dropdown of users

slider for Top-K

recommended movies table

**🎓 Educational Value**

This project helps understand:

collaborative filtering from scratch

latent vector learning

ranking metrics and tradeoffs

evaluation beyond RMSE

deploying a simple recommender UI

**🔮 Future Work (roadmap)**

You can extend this project with:

⏩ implicit feedback (clicks / views instead of ratings)

🎯 BPR / WARP ranking loss

🧭 cold-start handling with content features

🧠 hybrid recommender (metadata + MF)

🧪 cross-validation hyperparameter search

☁️ deployment on Streamlit Cloud / HuggingFace Spaces

🧩 neighborhood models + MF ensemble