# Train trên Kaggle (GPU)

Dataset không cần upload: Kaggle tự sinh lại bằng generator với cùng seed (khoảng 5 giây), nên dữ liệu giống hệt trên máy.

## Một lần chuẩn bị

1. Push repo `amc-qam-resnet` lên GitHub (public, hoặc private kèm token).
2. Kaggle → Settings → xác minh số điện thoại (bắt buộc để bật GPU và Internet).
3. Tạo Notebook mới → bên phải chọn **Accelerator: GPU T4 x2** (hoặc P100), **Internet: On**.

## Các cell cần chạy

```python
!git clone https://github.com/<github-user>/amc-qam-resnet.git
%cd amc-qam-resnet
!python -m dataset.generator          # data/train|val|test.npz, 120k bursts
!python -m pytest -q                  # 13 passed = dữ liệu và model đúng
!python train.py --all                # 3 model x 3 seed, log từng epoch
!python evaluate.py                   # hình + bảng kết quả vào results/
!zip -qr /kaggle/working/outputs.zip checkpoints results
```

Thời gian ước tính trên GPU: vài giây mỗi epoch, cả 9 run khoảng 15–30 phút (chưa đo thật, sẽ cập nhật sau lần chạy đầu).

## Lấy kết quả về máy

Tab **Output** bên phải → tải `outputs.zip` → giải nén vào `E:\emerging\amc-qam-resnet\` (ghi đè `checkpoints/` và `results/`).

## Nếu chỉ muốn chạy thử nhanh

```python
!python train.py --model resnet1d --seed 0 --epochs 3
```

## Lỗi thường gặp

- `CUDA not available`: quên bật Accelerator, hoặc chưa xác minh số điện thoại.
- `git clone` lỗi mạng: Internet đang Off.
- Hết quota GPU (30 giờ/tuần): chạy từng model riêng bằng `--model`.
