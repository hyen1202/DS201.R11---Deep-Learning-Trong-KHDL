# LeNet-5 · VinaFood21

```
LeNet/
├── data.py        # transform + DataLoader
├── model.py       # LeNet-5
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
%cd /content/LeNet
!pip install -r requirements.txt

!python model.py   # kiểm tra shape
!python train.py --data_root /content/VinaFood21 --pool avg --img_size 64 --batch_size 32 --epochs 20 --out_dir outputs/avg
!python train.py --data_root /content/VinaFood21 --pool max --img_size 64 --batch_size 32 --epochs 20 --out_dir outputs/max
!python evaluate.py --data_root /content/VinaFood21 --ckpt outputs/avg/best.pt
!python evaluate.py --data_root /content/VinaFood21 --ckpt outputs/max/best.pt
```

Tham số của `train.py`: `--img_size 64 --batch_size 32 --epochs 20 --lr 1e-3 --num_workers 2 --seed 42 --pool avg --out_dir outputs`.

Kết quả trong `outputs/`: `best.pt`, `history.csv`, `loss_curve.png`, `confusion_matrix.png`, `test_metrics.csv`, `classification_report.csv`.

