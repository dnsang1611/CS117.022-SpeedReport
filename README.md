# 1. Giới thiệu

Môn: Tư duy tính toántoán

Lớp: CS117.022

Năm học: 2023-2024

Đồ án: Vietnamese phone number extraction from Vietnamese sign boards

Thành viên:

|Họ và tên          | MSSV      |
--- | ---
| Đoàn Nhật Sang | 21522542 |
| Hoàng Quang Khải | 21520952 |
| Nguyễn Đình Minh Toàn  |21520486 |
| Nguyễn Thành Đạt | 21520705 |
| Trần Trung Tín |21522679|
# 2. Mở đầu 

Đây là source code để báo cáo về tốc độ chạy của DBNetpp(ResNet18) và PARSeq

# 3. Cài đặt môi trường

```
conda create -n aic2021 python=3.8
conda activate aic2021

git clone https://github.com/dnsang1611/CS117.022-SpeedReport.git

cd CS117.022-SpeedReport/parseq
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements/parseq.txt -e .[train,test]

cd ../mmocr
pip install -U openmim
mim install mmengine
mim install 'mmcv>==2.0.1'
mim install 'mmdet==3.1.0'
pip install -v -e .
```

# 4. Chuẩn bị dữ liệu

Chúng tôi sử dụng bộ dữ liệu VinText  được cung cấp bởi VinAI. Đầu tiên, hãy tải bộ dữ liệu này trong folder CS117.022-SpeedReport theo command sau:

```
gdown 1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml
unzip vietnamese_original.zip
rm vietnamese_original.zip
python prepare_data/crop_image.py
```

Các dữ liệu cần thiết sẽ nằm trong folder vietnamese

# 5. Download pretrained
```
mkdir pretrained
cd pretrained
gdown 1YXrUQVwpgMckN8OYxW6QZBSVSGzY5ubB
gdown 1XtKMsUHHJbOKMYuSco2QkvhhxN1OSJ3Z
cd ..

# 6. Speed report

```
python speed_report.py
```

# 7. Acknowledgement
Minghui Liao cùng đồng nghiệp với DBNetpp. \
Darwin Bautista, Rowel Atienza với PARSeq. \
VinAI với bộ dữ liệu VinText. \
MMOCR.
