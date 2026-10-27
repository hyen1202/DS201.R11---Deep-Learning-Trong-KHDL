import torch 
import torch.nn as nn

def conv_relu(in_ch, out_ch, k, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding),
        nn.ReLU(inplace=True))

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.b1 = conv_relu(in_ch, f1, 1)
        self.b2 = nn.Sequential(
            conv_relu(in_ch, f3r, 1),
            conv_relu(f3r, f3, 3, padding=1))
        self.b3 = nn.Sequential(
            conv_relu(in_ch, f5r, 1),
            conv_relu(f5r, f5, 5, padding=2))
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_ch, fp, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, in_ch=3, img=224, num_classes=21):
        super().__init__()

        self.c1 = conv_relu(in_ch, 64, 7, stride=2, padding=3)
        self.s2 = nn.MaxPool2d(3, 2, padding=1)
        self.c3 = conv_relu(64, 64, 1)
        self.c4 = conv_relu(64, 192, 3, padding=1)
        self.s5 = nn.MaxPool2d(3, 2, padding=1)

        self.inc6 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inc7 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.s8 = nn.MaxPool2d(3, 2, padding=1)
        self.inc9 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inc10 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)

        n_flat = self._flat_dim(in_ch, img)
        self.fc = nn.Linear(n_flat, num_classes)

    def _flat_dim(self, in_ch, img):
        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.gap(self.inc10(self.inc9(self.s8(self.inc7(self.inc6(self.s5(self.c4(self.c3(self.s2(self.c1(z)))))))))))
        return int(z.flatten(1).shape[1])

    def forward(self, x):
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.s5(x)
        x = self.inc6(x)
        x = self.inc7(x)
        x = self.s8(x)
        x = self.inc9(x)
        x = self.inc10(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.fc(x)

def n_params(module):
    return sum(p.numel() for p in module.parameters())

def layer_table(model, img, in_ch=3):
    names = ["c1", "s2", "c3", "c4", "s5", "inc6", "inc7", "s8", "inc9", "inc10", "gap"]
    x = torch.zeros(1, in_ch, img, img, device=next(model.parameters()).device)
    rows, total = [], 0
    with torch.no_grad():
        for name in names:
            layer = getattr(model, name)
            x = layer(x)
            p = n_params(layer)
            total += p
            rows.append((name.upper(), "x".join(map(str, tuple(x.shape)[1:])), p))
        x = x.flatten(1)
        rows.append(("flatten", str(x.shape[1]), 0))
        x = model.fc(x)
        p = n_params(model.fc)
        total += p
        rows.append(("FC", str(x.shape[1]), p))
    return rows, total

def inception_channels(rows):
    return [(name, int(shape.split("x")[0]))
            for name, shape, _ in rows if name.startswith("INC")]

if __name__ == "__main__":
    m = GoogLeNet(in_ch=3, img=64, num_classes=21)
    x = torch.rand(1, 3, 64, 64)
    print("input:", tuple(x.shape))
    print("logits:", tuple(m(x).shape))

    # in ra shape và tham số của từng layer
    rows, total = layer_table(m, 64)
    print(f"{'layer':<10}{'output':<16}{'tham số':>12}")
    for name, shape, p in rows:
        print(f"{name:<10}{shape:<16}{p:>12,}")
    print(f"{'TỔNG':<10}{'':<16}{total:>12,}")

    print("\nSố channel đầu ra của các Inception block:")
    for name, c in inception_channels(rows):
        print(f"{name:<10}{c:>6}")

    # lưu lại bảng shape và tham số
    import pandas as pd
    df = pd.DataFrame(rows, columns=["layer", "output", "params"])
    df.loc[len(df)] = ["TỔNG", "", total]
    df.to_csv("googlenet_params.csv", index=False, encoding="utf-8-sig")