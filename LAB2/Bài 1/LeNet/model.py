import torch 
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, in_ch=3, img=64, num_classes=21, pool="avg"):
        super().__init__()
        Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = Pool(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = Pool(2, 2)

        n_flat = self._flat_dim(in_ch, img)
        self.fc5 = nn.Linear(n_flat, 120)
        self.fc6 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, num_classes)

    def _flat_dim(self, in_ch, img):
        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.s4(self.c3(self.s2(self.c1(z))))
        return int(z.flatten(1).shape[1])

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = self.s2(x)
        x = torch.relu(self.c3(x))
        x = self.s4(x)
        x = x.flatten(1)
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x

def n_params(module):
    return sum(p.numel() for p in module.parameters())

def layer_table(model, img, in_ch=3):
    x = torch.zeros(1, in_ch, img, img)
    rows, total = [], 0
    for name, layer in [("C1", model.c1), ("S2", model.s2),
                        ("C3", model.c3), ("S4", model.s4)]:
        x = layer(x)
        p = n_params(layer)
        total += p
        rows.append((name, "x".join(map(str, tuple(x.shape)[1:])), p))
    x = x.flatten(1)
    rows.append(("flatten", str(x.shape[1]), 0))
    for name, layer in [("F5", model.fc5), ("F6", model.fc6), ("F7", model.fc7)]:
        x = layer(x)
        p = n_params(layer)
        total += p
        rows.append((name, str(x.shape[1]), p))
    return rows, total

if __name__ == "__main__":
    m = LeNet(in_ch=3, img=64, num_classes=21)
    x = torch.rand(1, 3, 64, 64)
    print("input:", tuple(x.shape))
    print("logits:", tuple(m(x).shape))

    # in ra shape và tham số của từng layer
    rows, total = layer_table(m, 64)
    print(f"{'layer':<10}{'output':<16}{'tham số':>12}")
    for name, shape, p in rows:
        print(f"{name:<10}{shape:<16}{p:>12,}")
    print(f"{'TỔNG':<10}{'':<16}{total:>12,}")

    # lưu lại bảng shape và tham số
    import pandas as pd
    df = pd.DataFrame(rows, columns=["layer", "output", "params"])
    df.loc[len(df)] = ["TỔNG", "", total]
    df.to_csv("lenet5_params.csv", index=False, encoding="utf-8-sig")
