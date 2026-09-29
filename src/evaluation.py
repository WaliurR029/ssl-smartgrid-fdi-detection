# src/evaluations.py

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import torch

def evaluate_fdi_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str = "../results",
    model_name: str = "scratch_baseline"
) -> dict:
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(y_batch.numpy().tolist())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(all_targets, all_preds)

    metrics = {
        "Model": model_name,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(auc),
        "Confusion_Matrix": cm.tolist()
    }

    print(f"\n=== Evaluation Report: {model_name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    metrics_dir = Path(save_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    fig_dir = Path(save_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'FDI Attack'], 
                yticklabels=['Normal', 'FDI Attack'])
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", color="darkorange")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_roc_curve.png", dpi=300)
    plt.close()

    return metrics