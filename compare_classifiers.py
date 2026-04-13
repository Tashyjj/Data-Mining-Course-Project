import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# loading data
DATA_PATH = "creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())
print("\nClass distribution:")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))


# =========================================================
# 2.features and labels for the table
X = df.drop("Class", axis=1)
y = df["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)


# =========================================================
# 3. downsampling for speed
USE_SUBSET = True

if USE_SUBSET:
    # keeping all fraud cases in training subset, and sample non-fraud cases
    train_df = X_train.copy()
    train_df["Class"] = y_train.values

    fraud_train = train_df[train_df["Class"] == 1]
    nonfraud_train = train_df[train_df["Class"] == 0]

    nonfraud_sample = nonfraud_train.sample(n=5000, random_state=42)

    train_subset = pd.concat([fraud_train, nonfraud_sample], axis=0)
    train_subset = train_subset.sample(frac=1, random_state=42)  # everyday im shuffling

    X_train_used = train_subset.drop("Class", axis=1)
    y_train_used = train_subset["Class"]

    print("\nUsing training subset for speed:")
    print(X_train_used.shape)
    print(y_train_used.value_counts())
else:
    X_train_used = X_train
    y_train_used = y_train


# =========================================================
# 4. defininf the models

log_reg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf", probability=True, random_state=42))
])

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())
])


# =========================================================
# 5. cross validation seciton

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

knn_param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11]
}

svm_param_grid = {
    # gamma is related to sigma in the RBF kernel
    "model__C": [0.1, 1, 10],
    "model__gamma": ["scale", 0.01, 0.1, 1]
}

print("\nRunning KNN cross-validation...")
knn_cv_start = time.time()
knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
knn_grid.fit(X_train_used, y_train_used)
knn_cv_end = time.time()

print("Best KNN params:", knn_grid.best_params_)
print("Best KNN CV ROC-AUC:", knn_grid.best_score_)
print("KNN CV time (s):", round(knn_cv_end - knn_cv_start, 4))


print("\nRunning SVM cross-validation...")
svm_cv_start = time.time()
svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)
svm_grid.fit(X_train_used, y_train_used)
svm_cv_end = time.time()

print("Best SVM params:", svm_grid.best_params_)
print("Best SVM CV ROC-AUC:", svm_grid.best_score_)
print("SVM CV time (s):", round(svm_cv_end - svm_cv_start, 4))


# regression model can be fit directly without parameter tuning for starter code.
best_log_reg = log_reg_pipeline
best_knn = knn_grid.best_estimator_
best_svm = svm_grid.best_estimator_


# =========================================================
# 6. training and timing
results = {}

models = {
    "Logistic Regression": best_log_reg,
    "K-Nearest Neighbors": best_knn,
    "Support Vector Machine (RBF)": best_svm
}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    train_start = time.time()
    model.fit(X_train_used, y_train_used)
    train_end = time.time()

    train_time = train_end - train_start
    results[model_name] = {
        "model": model,
        "train_time": train_time
    }

    print(f"{model_name} training time: {train_time:.4f} seconds")


# =========================================================
# 7. testing and timing
for model_name, info in results.items():
    model = info["model"]

    print(f"\nTesting {model_name}...")
    test_start = time.time()

    y_pred = model.predict(X_test)

    # need score/probabilities for ROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    test_end = time.time()

    test_time = test_end - test_start
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    info["test_time"] = test_time
    info["y_pred"] = y_pred
    info["y_score"] = y_score
    info["confusion_matrix"] = cm
    info["fpr"] = fpr
    info["tpr"] = tpr
    info["roc_auc"] = roc_auc

    print(f"{model_name} testing time: {test_time:.4f} seconds")
    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    print(f"{model_name} confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, digits=4))


# =========================================================
# 8. plotting curves
plt.figure(figsize=(8, 6))

for model_name, info in results.items():
    plt.plot(info["fpr"], info["tpr"], label=f"{model_name} (AUC = {info['roc_auc']:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Credit Card Fraud Classification")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================================================
# 9. confusion matrices
for model_name, info in results.items():
    disp = ConfusionMatrixDisplay(confusion_matrix=info["confusion_matrix"])
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()


# =========================================================
# 10. summary
summary_rows = []

for model_name, info in results.items():
    summary_rows.append({
        "Algorithm": model_name,
        "Training Time (s)": round(info["train_time"], 4),
        "Testing Time (s)": round(info["test_time"], 4),
        "ROC-AUC": round(info["roc_auc"], 4)
    })

summary_df = pd.DataFrame(summary_rows)
print("\nSummary of Results:")
print(summary_df)
summary_df.to_csv("classifier_results_summary.csv", index=False)
print("\nSaved summary to classifier_results_summary.csv")