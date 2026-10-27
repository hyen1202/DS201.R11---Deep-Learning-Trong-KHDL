# ResNet-18 · VinaFood21

```
ResNet18/
├── data.py        # transform + DataLoader
├── model.py       # ResNet-18
├── engine.py      # train_one_epoch, evaluate
├── train.py       # chạy huấn luyện, lưu outputs/best.pt
└── evaluate.py    # đánh giá trên test, lưu confusion matrix
```

Dữ liệu theo cấu trúc `ImageFolder`; không có `val/` thì tự tách 10% từ `train/`:

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

## Chạy trên Colab (Terminal)

```bash
%cd /content/ResNet18
!pip install -r requirements.txt

!python model.py   # kiểm tra shape
!python train.py --data_root /content/VinaFood21 --img_size 112 --batch_size 32 --epochs 20 --seed 42 --out_dir outputs/aug
!python train.py --data_root /content/VinaFood21 --img_size 112 --batch_size 32 --epochs 20 --seed 42 --no_aug --out_dir outputs/noaug
!python evaluate.py --data_root /content/VinaFood21 --ckpt outputs/aug/best.pt
!python evaluate.py --data_root /content/VinaFood21 --ckpt outputs/noaug/best.pt
```

Tham số của `train.py`: `--img_size 112 --batch_size 32 --epochs 20 --lr 1e-3 --num_workers 2 --seed 42 --no_aug --out_dir outputs`.

- `--no_aug`: tắt data augmentation (bỏ cờ này thì train có augmentation, đây là mặc định).

Kết quả của mỗi lần chạy nằm trong thư mục `--out_dir` (ví dụ `outputs/aug/`, `outputs/noaug/`): `best.pt`, `history.csv`, `loss_curve.png`, `confusion_matrix.png`, `test_metrics.csv`, `classification_report.csv`.

