"""ResNet-50 pretrained (HuggingFace) cho Bài 4*: PretrainedResnet ở B.6 Bước 2-3."""
import torch
import torch.nn as nn
from transformers import ResNetForImageClassification

MODEL_NAME = "microsoft/resnet-50"


class PretrainedResnet(nn.Module):
    """Backbone ResNet-50 pretrained + classifier mới nn.Linear(2048, num_classes)."""

    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()
        base = ResNetForImageClassification.from_pretrained(MODEL_NAME)
        self.resnet = base.resnet                                  # bỏ classifier 1000 lớp của ImageNet
        self.classifier = nn.Linear(2048, num_classes, bias=True)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:                    # feature extraction: chỉ học classifier
            for p in self.resnet.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:               # requires_grad=False không chặn BatchNorm cập nhật
            self.resnet.eval()                 # running stats, nên giữ backbone ở chế độ eval
        return self

    def forward(self, images):
        features = self.resnet(images).pooler_output    # [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)     # [B, 2048]
        return self.classifier(features)                # [B, 21]


def n_params(module, trainable_only=False):
    ps = module.parameters()
    if trainable_only:
        ps = (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)


if __name__ == "__main__":
    import pandas as pd

    fe = PretrainedResnet(21, freeze_backbone=True)
    ft = PretrainedResnet(21, freeze_backbone=False)

    # shape từ ảnh vào đến logits
    ft.eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        feats = ft.resnet(x).pooler_output
        logits = ft(x)
    print("backbone      :", f"{n_params(ft.resnet):,}", "tham số")
    print("ảnh vào       :", tuple(x.shape))
    print("pooler_output :", tuple(feats.shape))
    print("sau squeeze   :", tuple(feats.squeeze(-1).squeeze(-1).shape))
    print("logits        :", tuple(logits.shape), "\n")
    assert feats.shape == (2, 2048, 1, 1) and logits.shape == (2, 21)

    # số tham số được cập nhật ở từng chế độ
    rows = [("feature extraction", n_params(fe), n_params(fe, True), "1e-3"),
            ("fine-tuning", n_params(ft), n_params(ft, True), "1e-4")]
    print(f"{'chế độ':<22}{'tổng':>14}{'được cập nhật':>16}{'lr':>8}")
    for name, total, trainable, lr in rows:
        print(f"{name:<22}{total:>14,}{trainable:>16,}{lr:>8}")
    pd.DataFrame(rows, columns=["mode", "total", "trainable", "lr"]).to_csv(
        "resnet50_params.csv", index=False, encoding="utf-8-sig")

    assert n_params(fe) == 23_551_061
    assert n_params(fe, trainable_only=True) == 43_029      # chỉ classifier
    assert n_params(ft, trainable_only=True) == 23_551_061

    # optimizer chỉ nhận tham số requires_grad (B.6 Bước 3)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, fe.parameters()), lr=1e-3)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print("tham số trong optimizer (feature extraction):", f"{n_opt:,}")
    assert n_opt == 43_029
