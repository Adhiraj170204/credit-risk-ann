"""Credit default prediction: Logistic Regression baseline vs a Keras ANN.

Steps
1. Stratified 70/15/15 split, preprocessing fitted on train only.
2. Baseline: Logistic Regression (class_weight="balanced").
3. ANN: three small architectures, each trained with Adam, class weights,
   dropout and early stopping on validation loss. Best one picked on validation ROC-AUC.
4. Decision threshold picked on the validation set (max F1), never on test.
5. Final report on the untouched test set.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

from prepare import SEED, load_clean, make_preprocessor, split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

ARCHITECTURES = {
    "ann_small  [16]":      [16],
    "ann_medium [32-16]":   [32, 16],
    "ann_large  [64-32-16]": [64, 32, 16],
}
DROPOUT = 0.3
L2 = 1e-4
LR = 1e-3
BATCH = 64
MAX_EPOCHS = 300
PATIENCE = 20


def best_f1_threshold(y_true, p):
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(name, y_true, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "model": name, "threshold": round(thr, 3),
        "roc_auc": roc_auc_score(y_true, p), "pr_auc": average_precision_score(y_true, p),
        "accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def build_ann(n_features, hidden):
    model = tf.keras.Sequential([tf.keras.Input(shape=(n_features,))])
    for units in hidden:
        model.add(tf.keras.layers.Dense(units, activation="relu",
                                        kernel_regularizer=tf.keras.regularizers.l2(L2)))
        model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
    return model


def main():
    df = load_clean()
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(df)
    pre = make_preprocessor().fit(X_tr)
    A_tr, A_val, A_te = (pre.transform(x).astype("float32") for x in (X_tr, X_val, X_te))
    print(f"train {A_tr.shape}, val {A_val.shape}, test {A_te.shape}")

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight = {0: weights[0], 1: weights[1]}

    # ---------- Baseline ----------
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(A_tr, y_tr)
    lr_val, lr_te = lr.predict_proba(A_val)[:, 1], lr.predict_proba(A_te)[:, 1]
    val_rows = [{"model": "logistic_regression", "val_roc_auc": roc_auc_score(y_val, lr_val),
                 "params": A_tr.shape[1] + 1, "epochs": None}]

    # ---------- ANN architecture search (on validation) ----------
    runs = {}
    for name, hidden in ARCHITECTURES.items():
        tf.keras.utils.set_random_seed(SEED)
        model = build_ann(A_tr.shape[1], hidden)
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                restore_best_weights=True)
        hist = model.fit(A_tr, y_tr.values, validation_data=(A_val, y_val.values),
                         epochs=MAX_EPOCHS, batch_size=BATCH, class_weight=class_weight,
                         callbacks=[stop], verbose=0)
        p_val = model.predict(A_val, verbose=0).ravel()
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        runs[name] = (model, hist, p_val)
        val_rows.append({"model": name.strip(), "val_roc_auc": roc_auc_score(y_val, p_val),
                         "params": model.count_params(), "epochs": best_epoch})
        print(f"{name:24s} val AUC {val_rows[-1]['val_roc_auc']:.4f}  "
              f"best epoch {best_epoch}  params {model.count_params()}")

    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(RESULTS / "validation_comparison.csv", index=False)
    best_name = max(runs, key=lambda n: roc_auc_score(y_val, runs[n][2]))
    model, hist, ann_val = runs[best_name]
    ann_te = model.predict(A_te, verbose=0).ravel()
    print("Best ANN on validation:", best_name.strip())

    # ---------- Test-set report (thresholds chosen on validation) ----------
    test = pd.DataFrame([
        evaluate("logistic_regression", y_te, lr_te, best_f1_threshold(y_val, lr_val)),
        evaluate(best_name.strip(), y_te, ann_te, best_f1_threshold(y_val, ann_val)),
    ])
    test.to_csv(RESULTS / "test_metrics.csv", index=False)
    pd.set_option("display.width", 200)
    print(test.round(3).to_string(index=False))

    with open(RESULTS / "best_ann.json", "w") as f:
        json.dump({"architecture": best_name.strip(), "hidden_layers": ARCHITECTURES[best_name],
                   "dropout": DROPOUT, "l2": L2, "learning_rate": LR, "batch_size": BATCH,
                   "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
                   "params": model.count_params(), "n_features": A_tr.shape[1],
                   "class_weight": {k: round(float(v), 3) for k, v in class_weight.items()}},
                  f, indent=2)
    model.save(RESULTS / "best_ann.keras")

    # ---------- Learning curves ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="validation")
    ax[0].axvline(np.argmin(hist.history["val_loss"]), ls="--", c="grey", label="best epoch")
    ax[0].set(title="Loss (binary cross-entropy)", xlabel="epoch"); ax[0].legend()
    ax[1].plot(hist.history["auc"], label="train"); ax[1].plot(hist.history["val_auc"], label="validation")
    ax[1].set(title="ROC-AUC", xlabel="epoch"); ax[1].legend()
    fig.suptitle(f"Training curves: {best_name.strip()}")
    fig.tight_layout(); fig.savefig(RESULTS / "figures" / "learning_curves.png", dpi=120)


if __name__ == "__main__":
    main()
