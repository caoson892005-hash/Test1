# Social VLM Camera

MVP dùng webcam để:

1. Phát hiện class `person` bằng YOLO chạy cục bộ.
2. In `Đã tìm thấy người!` khi một người xuất hiện.
3. Tạo vùng xã hội ứng viên khi hai người đứng gần nhau.
4. Gửi riêng vùng crop sang LLaVA chạy cục bộ bằng Ollama.

> Phiên bản này dùng khoảng cách hình học để tạo **ứng viên** vùng xã hội.
> Một ảnh đơn không đủ chứng minh chắc chắn hai người đang nói chuyện. Bản tiếp
> theo nên thêm tracking, hướng mặt, âm thanh và active-speaker detection.

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
cd /home/son/social_vlm_camera
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ultralytics sẽ tải `yolo11n.pt` trong lần chạy đầu nếu máy chưa có model.

## Cài Ollama và LLaVA

Sau khi cài [Ollama](https://ollama.com/download), tải model:

```bash
ollama pull llava:7b
```

Kiểm tra model đã có:

```bash
ollama list
```

Mặc định chương trình dùng `llava:7b`. Có thể chọn model khác đã tải:

```bash
export OLLAMA_MODEL="llava:13b"
```

Không cần OpenAI API key. Ảnh được xử lý cục bộ và không gửi lên OpenAI.

## Chạy

Chỉ kiểm tra camera và phát hiện người, chưa gọi VLM:

```bash
python app.py --no-vlm
```

Chạy đầy đủ:

```bash
python app.py
```

Nếu camera mặc định không mở được:

```bash
python app.py --camera 1
```

Nhấn `q` để thoát.

Các tùy chọn quan trọng:

```bash
python app.py \
  --confidence 0.5 \
  --social-distance 1.8 \
  --vlm-interval 8
```

- `--confidence`: ngưỡng nhận diện người.
- `--social-distance`: khoảng cách chuẩn hóa để ghép hai người.
- `--vlm-interval`: giới hạn tần suất chạy LLaVA, giúp giảm tải tài nguyên.

## Luồng xử lý

```text
Webcam
  → YOLO person detector
  → lọc cặp người theo khoảng cách
  → crop social region
  → resize + JPEG compression
  → LLaVA qua Ollama (background thread)
```

LLaVA không được gọi trên mọi frame. Camera tiếp tục chạy trong khi luồng nền
đợi kết quả từ Ollama.

## Lỗi thường gặp

- `connection refused`: Ollama chưa chạy. Mở ứng dụng Ollama hoặc chạy
  `ollama serve` trong một terminal khác.
- `model not found`: chạy `ollama pull llava:7b`.
- Xử lý quá chậm: tăng `--vlm-interval`, dùng `llava:7b`, hoặc chạy với GPU.
