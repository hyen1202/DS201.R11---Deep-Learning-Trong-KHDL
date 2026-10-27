import torch 
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:       
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)             
        return torch.relu(out) 

class ResNet18(nn.Module):
    def __init__(self, in_ch=3, img=112, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = self._stage(64, 64, stride=1)
        self.layer2 = self._stage(64, 128, stride=2)
        self.layer3 = self._stage(128, 256, stride=2)
        self.layer4 = self._stage(256, 512, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(in_ch, out_ch, stride):
        return nn.Sequential(BasicBlock(in_ch, out_ch, stride),
                             BasicBlock(out_ch, out_ch, 1))

    def forward(self, x):
        x = self.stem(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.fc(self.gap(x).flatten(1))

def n_params(module):
    return sum(p.numel() for p in module.parameters())

def layer_table(model, img, in_ch=3):
    was_training = model.training
    model.eval()                                   
    x = torch.zeros(1, in_ch, img, img, device=next(model.parameters()).device)
    rows = []
    with torch.no_grad():
        for name in ["stem", "layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, name)
            x = layer(x)
            rows.append((name, tuple(x.shape), n_params(layer)))
        x = model.gap(x).flatten(1)
        rows.append(("avgpool", tuple(x.shape), n_params(model.gap)))
        x = model.fc(x)
        rows.append(("fc", tuple(x.shape), n_params(model.fc)))
    model.train(was_training)
    return rows, n_params(model)

if __name__ == "__main__":
    m = ResNet18(in_ch=3, img=112, num_classes=21)
    x = torch.rand(1, 3, 112, 112)
    print("input:", tuple(x.shape))
    print("logits:", tuple(m(x).shape))

    # in ra shape và tham số của từng layer
    rows, total = layer_table(m, 112)
    print(f"{'stage':<10}{'output shape':<24}{'tham số':>12}")
    for name, shape, p in rows:
        print(f"{name:<10}{str(shape):<24}{p:>12,}")
    print(f"{'TỔNG':<10}{'':<24}{total:>12,}")

    import pandas as pd
    df = pd.DataFrame([(n, str(s), p) for n, s, p in rows],
                      columns=["stage", "output shape", "params"])
    df.loc[len(df)] = ["TỔNG", "", total]
    df.to_csv("resnet18_params.csv", index=False, encoding="utf-8-sig")