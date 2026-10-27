# Bài 4* · ResNet-50 pretrained · VinaFood21

```
ResNet50/
├── data.py                # transform (224×224, mean/std ImageNet) + DataLoader
├── pretrained_resnet.py   # PretrainedResnet (HuggingFace microsoft/resnet-50 + nn.Linear(2048, 21))
├── engine.py              # train_one_epoch, evaluate (trả loss, acc, f1, y_true, y_pred)
├── train.py               # --mode fe | ft, lưu <out_dir>/best.pt, history.csv, loss_curve.png
├── evaluate.py            # đánh giá test, lưu test_metrics.csv, classification_report.csv, confusion_matrix.png
```

Dữ liệu theo cấu trúc `ImageFolder`; không có `val/` thì tự tách 10% từ `train/` (seed 42).
Dùng cùng `data.py` (cùng phép chia, cùng seed) với các bài trước để so sánh công bằng.

```
VinaFood21/
├── train/<tên lớp>/*.jpg
├── val/<tên lớp>/*.jpg     (tuỳ chọn)
└── test/<tên lớp>/*.jpg
```
Nếu tải dữ liệu trên Colab:
```
!pip -q install gdown
!gdown 1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR -O /content/vinafood21.zip
!unzip -q /content/vinafood21.zip -d /content/VinaFood21
!mv /content/VinaFood21/VinaFood21/* /content/VinaFood21/
!rmdir /content/VinaFood21/VinaFood21
!ls /content/VinaFood21
```

## Chạy trên Colab

Cần bật Internet để tải `microsoft/resnet-50` (khoảng 100 MB) từ HuggingFace.

```bash
%cd /content/ResNet50
!pip install -r requirements.txt

!python pretrained_resnet.py   # kiểm tra shape, 23,551,061 / 43,029 / 23,551,061 tham số
!python train.py --data_root /content/VinaFood21 --mode fe --out_dir outputs/fe
!python train.py --data_root /content/VinaFood21 --mode ft --out_dir outputs/ft
!python evaluate.py --data_root /content/VinaFood21 --ckpt outputs/fe/best.pt
!python evaluate.py --data_root /content/VinaFood21 --ckpt outputs/ft/best.pt
```

Tham số của `train.py`: `--mode ft --img_size 224 --batch_size 32 --epochs 20 --lr (theo mode) --num_workers 2 --seed 42 --out_dir outputs`.

- `--mode fe`: feature extraction, đóng băng backbone, chỉ học classifier (43,029 tham số), Adam lr 1e-3.
- `--mode ft`: fine-tuning toàn mạng (23,551,061 tham số), Adam lr 1e-4.
- Optimizer chỉ nhận tham số `requires_grad`; ở chế độ `fe` backbone được giữ ở `eval()` để BatchNorm không đổi.

Mỗi chế độ ghi vào `--out_dir`: `best.pt`, `history.csv`, `loss_curve.png` (đánh dấu epoch hai đường loss tách nhau), `confusion_matrix.png`, `test_metrics.csv`, `classification_report.csv`. `summarize.py` ghi thêm `summary_resnet50.csv` và `f1_per_class_fe_vs_ft.csv`.


