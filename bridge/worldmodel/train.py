#!/usr/bin/env python3
"""
train.py — train the dynamics MLP (state, action) -> [delta_energy, displacement]
on transitions built by dataset.py. CPU-only on purpose: the net is tiny and
this must never touch Delphi's VRAM.

Outputs (all in bridge/worldmodel/):
  model.pt      state_dict + normalization + config (deterministic inference)
  metrics.json  val/test MAE per target + compounding rollout error (K=1,3,5,10)
  plots/pred_vs_actual.png, plots/rollout_error.png
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
HIDDEN = 128
EPOCHS = 300
PATIENCE = 25
BATCH = 256


class DynamicsMLP(nn.Module):
    def __init__(self, n_in=13, n_out=2, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def load_data():
    d = np.load(os.path.join(HERE, "data.npz"), allow_pickle=True)
    with open(os.path.join(HERE, "norm.json")) as f:
        norm = json.load(f)
    return d, norm


def standardize(a, mean, std):
    return (a - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


def rollout_error(model, d, norm, horizons=(1, 3, 5, 10)):
    """Autoregressive energy rollout on test episodes: predicted energy is fed
    back as the energy feature; terrain/speed features stay as actually logged
    (terrain along the true path is known). Reports mean |E_pred - E_true| per
    horizon, plus cumulative-displacement error."""
    X, Y, eps, ts = d["X_test"], d["Y_test"], d["seq_test"], d["t_test"]
    xm, xs = np.asarray(norm["x_mean"]), np.asarray(norm["x_std"])
    ym, ys = np.asarray(norm["y_mean"]), np.asarray(norm["y_std"])
    err = {k: [] for k in horizons}
    disp_err = {k: [] for k in horizons}
    maxK = max(horizons)
    for ep in np.unique(eps):
        idx = np.where(eps == ep)[0]          # rows are stored in time order per agent-sequence
        for s in range(len(idx)):
            window = idx[s:s + maxK]
            if len(window) < 1:
                continue
            # truncate at time gaps (excluded eat transitions break continuity)
            contiguous = [window[0]]
            for a, b in zip(window, window[1:]):
                if not (1.5 <= ts[b] - ts[a] <= 2.5):
                    break
                contiguous.append(b)
            window = np.asarray(contiguous)
            e_pred = X[window[0], 0]
            e_true = X[window[0], 0]
            cum_pred, cum_true = 0.0, 0.0
            for k, row in enumerate(window, start=1):
                x = X[row].copy()
                e_true = X[row, 0]
                x[0] = e_pred                  # feed predicted energy back
                xn = torch.tensor(standardize(x[None], xm, xs), dtype=torch.float32)
                with torch.no_grad():
                    yn = model(xn).numpy()[0]
                y = yn * ys + ym
                e_pred = e_pred + y[0]
                cum_pred += y[1]
                cum_true += Y[row, 1]
                actual_next_e = X[row, 0] + Y[row, 0]
                if k in err:
                    err[k].append(abs(e_pred - actual_next_e))
                    disp_err[k].append(abs(cum_pred - cum_true))
    return ({k: float(np.mean(v)) for k, v in err.items() if v},
            {k: float(np.mean(v)) for k, v in disp_err.items() if v})


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    d, norm = load_data()
    Xtr = standardize(d["X_train"], norm["x_mean"], norm["x_std"])
    Ytr = standardize(d["Y_train"], norm["y_mean"], norm["y_std"])
    Xva = standardize(d["X_val"], norm["x_mean"], norm["x_std"])
    Yva = standardize(d["Y_val"], norm["y_mean"], norm["y_std"])

    model = DynamicsMLP(Xtr.shape[1], Ytr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xva_t, Yva_t = torch.tensor(Xva), torch.tensor(Yva)

    best_val, best_state, bad = float("inf"), None, 0
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            i = perm[b:b + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[i]), Ytr_t[i])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = lossf(model(Xva_t), Yva_t).item()
        if val < best_val - 1e-5:
            best_val, best_state, bad = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()

    # physical-unit MAE
    ys = np.asarray(norm["y_std"]); ym = np.asarray(norm["y_mean"])
    def mae(X, Y):
        with torch.no_grad():
            p = model(torch.tensor(standardize(X, norm["x_mean"], norm["x_std"]))).numpy()
        p = p * ys + ym
        return np.abs(p - Y).mean(0).tolist(), p
    val_mae, _ = mae(d["X_val"], d["Y_val"])
    test_mae, test_pred = mae(d["X_test"], d["Y_test"])
    roll_e, roll_d = rollout_error(model, d, norm)

    metrics = {
        "epochs_trained": epoch + 1,
        "counts": norm["counts"],
        "val_mae": dict(zip(norm["target_names"], val_mae)),
        "test_mae": dict(zip(norm["target_names"], test_mae)),
        "rollout_energy_mae_kJ_by_horizon": roll_e,
        "rollout_displacement_mae_m_by_horizon": roll_d,
        "target_std_test": np.std(d["Y_test"], 0).tolist(),
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    torch.save({"state_dict": model.state_dict(), "norm": norm,
                "config": {"hidden": HIDDEN, "n_in": Xtr.shape[1], "n_out": Ytr.shape[1], "seed": SEED}},
               os.path.join(HERE, "model.pt"))

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(norm["target_names"]):
        ax = axes[i]
        ax.scatter(d["Y_test"][:, i], test_pred[:, i], s=3, alpha=0.3)
        lo, hi = d["Y_test"][:, i].min(), d["Y_test"][:, i].max()
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"actual {name}"); ax.set_ylabel(f"predicted {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", "pred_vs_actual.png"), dpi=110)
    fig2, ax = plt.subplots(figsize=(5, 4))
    ks = sorted(roll_e)
    ax.plot(ks, [roll_e[k] for k in ks], "o-", label="energy MAE (kJ)")
    ax.plot(ks, [roll_d[k] for k in ks], "s-", label="cum. displacement MAE (m)")
    ax.set_xlabel("rollout horizon K (steps of ~2 s)"); ax.set_ylabel("MAE")
    ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, "plots", "rollout_error.png"), dpi=110)

    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
