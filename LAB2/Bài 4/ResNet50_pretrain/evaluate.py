import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, precision_recall_fscore_support)
from data import build_test_loader
from engine import evaluate
from pretrained_resnet import PretrainedResnet

def get_args():
    p = argparse.ArgumentParser(description="Đánh giá ResNet-50 pretrained trên tập test")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="outputs/best.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()

def save_confusion_matrix(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật")
    fig.colorbar(im)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    classes, img_size = ckpt["classes"], ckpt["img_size"]

    # trọng số pretrained chỉ để dựng kiến trúc, sau đó bị load_state_dict ghi đè
    model = PretrainedResnet(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model"])

    test_dl, test_classes = build_test_loader(
        args.data_root, img_size, args.batch_size, args.num_workers)
    assert test_classes == classes, "thứ tự lớp ở test khác train"

    loss, acc, f1, y_true, y_pred = evaluate(model, test_dl, nn.CrossEntropyLoss(), device)
    print(f"checkpoint epoch {ckpt['epoch']} ({ckpt.get('mode', '?')}) | test loss {loss:.4f}\n")

    out_dir = Path(args.ckpt).parent

    # tính các độ đo
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = pd.DataFrame([{
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1_macro": f1,
    }])
    
    print(f" {'-'*10} Kết quả trên test {'-'*10}")
    print(metrics.round(4).to_string(index=False), "\n")
    metrics.to_csv(out_dir / "test_metrics.csv", index=False)

    # F1 từng lớp: in ra + lưu csv
    rep = classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))
    pd.DataFrame(rep).T.to_csv(out_dir / "classification_report.csv", encoding="utf-8-sig")

    cm_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix(confusion_matrix(y_true, y_pred), classes, cm_path)
    print("đã lưu:", out_dir / "test_metrics.csv", "|",
          out_dir / "classification_report.csv", "|", cm_path)


if __name__ == "__main__":
    main()
