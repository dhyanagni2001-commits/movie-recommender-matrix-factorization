import numpy as np


class MatrixFactorization:

    def __init__(
        self,
        n_users,
        n_items,
        n_factors=40,
        lr=0.01,
        reg=0.1,
        n_epochs=20,
        random_state=42,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

        self.rng = np.random.default_rng(random_state)

        self.user_factors = None
        self.item_factors = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = 0.0

    def _init_params(self):
        self.user_factors = 0.01 * self.rng.standard_normal((self.n_users, self.n_factors))
        self.item_factors = 0.01 * self.rng.standard_normal((self.n_items, self.n_factors))
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)

    def _predict_internal(self, u, i):
        pu = self.user_factors[u]
        qi = self.item_factors[i]

        return float(
            self.global_mean
            + self.user_bias[u]
            + self.item_bias[i]
            + np.dot(pu, qi)
        )

    def fit(self, train_data):
        train_data = np.asarray(train_data)

        user_ids = train_data[:, 0].astype(int)
        item_ids = train_data[:, 1].astype(int)
        ratings = train_data[:, 2].astype(float)

        self.global_mean = ratings.mean()
        self._init_params()

        n_samples = ratings.shape[0]

        for epoch in range(1, self.n_epochs + 1):

            indices = np.arange(n_samples)
            self.rng.shuffle(indices)

            for idx in indices:
                u = user_ids[idx]
                i = item_ids[idx]
                r_ui = ratings[idx]

                r_hat = self._predict_internal(u, i)
                err = r_ui - r_hat

                pu = self.user_factors[u]
                qi = self.item_factors[i]

                # update biases
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

                # update latent factors
                self.user_factors[u] += self.lr * (err * qi - self.reg * pu)
                self.item_factors[i] += self.lr * (err * pu - self.reg * qi)

            print(f"Epoch {epoch:02d} completed")

    def predict_for_user(self, user_idx):
        pu = self.user_factors[user_idx]
        scores = (
            self.global_mean
            + self.user_bias[user_idx]
            + self.item_bias
            + self.item_factors @ pu
        )
        return scores

    def recommend_for_user(self, user_idx, known_items=None, top_k=10):
        scores = self.predict_for_user(user_idx)

        if known_items:
            scores = scores.copy()
            scores[list(known_items)] = -1e9

        top_idx = np.argsort(-scores)[:top_k]
        return top_idx, scores[top_idx]
