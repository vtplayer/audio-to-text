# Audio Transcription Tool

Công cụ chuyển đổi file audio thành văn bản kèm timestamp và phân biệt người nói. Hỗ trợ xử lý nhiều định dạng audio phổ biến và hiển thị tiến trình xử lý trực quan.

## Tính năng

- Chuyển đổi audio thành văn bản với thời gian chính xác
- Phân biệt và gắn nhãn từng người nói
- Hỗ trợ nhiều định dạng audio (MP3, WAV, M4A, AAC)
- Hiển thị tiến trình xử lý realtime
- Tối ưu hóa audio tự động
- Hỗ trợ tiếng Việt
- Logging chi tiết

## Yêu cầu hệ thống

- Python 3.9 trở lên
- FFmpeg
- Tài khoản Hugging Face

## Cài đặt

1. Cài đặt Python và pip:
```bash
# macOS
brew install python@3.9

# Ubuntu
sudo apt update
sudo apt install python3.9 python3-pip
```

2. Cài đặt FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

3. Tạo và kích hoạt môi trường ảo:
```bash
# Tạo môi trường
python3.9 -m venv venv

# Kích hoạt môi trường
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

4. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

## File requirements.txt
```
python-dotenv
SpeechRecognition
pydub
pyannote.audio
torch
torchaudio
transformers
soundfile
```

## Cấu hình Hugging Face

1. Đăng ký và accept license:
- Tạo tài khoản tại https://huggingface.co
- Accept license tại:
  - https://huggingface.co/pyannote/speaker-diarization
  - https://huggingface.co/pyannote/segmentation

2. Tạo access token:
- Truy cập https://huggingface.co/settings/tokens
- Tạo token mới với quyền read

3. Thiết lập token:
```bash
# Tạo file .env
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

## Cấu trúc thư mục
```
.
├── transcribe.py     # Script chính
├── requirements.txt  # Danh sách thư viện
├── .env             # Chứa token
├── logs/            # Chứa log files
└── temp/            # Thư mục tạm
```

## Sử dụng

1. Xử lý file MP3:
```bash
python transcribe.py input.mp3 -o output.txt
```

2. Xử lý các định dạng khác:
```bash
python transcribe.py input.wav -o output.txt
python transcribe.py input.m4a -o output.txt
python transcribe.py input.aac -o output.txt
```

3. Chế độ verbose:
```bash
python transcribe.py input.mp3 -o output.txt -v
```

### Tham số

- `input`: Đường dẫn file audio (MP3/WAV/M4A/AAC)
- `-o, --output`: Đường dẫn file kết quả đầu ra
- `-v, --verbose`: Hiển thị thông tin chi tiết

## Kết quả đầu ra

File kết quả có định dạng:
```
[00:00:05] SPEAKER_01: Nội dung người thứ nhất nói
[00:00:10] SPEAKER_02: Nội dung người thứ hai nói
...
```

## Progress Bar

Trong quá trình xử lý, công cụ hiển thị:
```
=== TIẾN TRÌNH XỬ LÝ ===
[===================>        ] 45.2%
Đoạn: 23/51
Thời gian: 00:05:30
Trạng thái: Đang xử lý đoạn 23/51
```

## Xử lý audio

1. Tối ưu hóa tự động:
- Chuyển stereo thành mono
- Chuẩn hóa âm lượng
- Resampling về 16kHz
- Tối ưu bitrate

2. Xử lý lỗi:
- Tự động retry khi gặp lỗi API
- Dọn dẹp file tạm
- Log chi tiết quá trình xử lý

## Xử lý lỗi phổ biến

1. Lỗi FFmpeg:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

2. Lỗi SSL:
```bash
pip install --upgrade certifi
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

3. Lỗi token:
- Kiểm tra file .env
- Verify token trên Hugging Face
- Accept lại license các model

## Tips sử dụng

1. Chất lượng audio:
- File audio chất lượng cao cho kết quả tốt hơn
- Tránh nhiễu và echo trong audio
- Tối ưu nhất là audio 16kHz, mono

2. Tài nguyên:
- Cần khoảng 4GB RAM cho xử lý
- Đảm bảo đủ dung lượng ổ cứng (khoảng 2GB cho models)
- Kết nối internet ổn định cho API calls

3. Debug:
- Sử dụng flag -v để xem log chi tiết
- Kiểm tra logs/ để xem lịch sử xử lý
- Đảm bảo đã accept license đầy đủ

## License

MIT License

## Support

Nếu gặp vấn đề:
1. Kiểm tra logs
2. Chạy với -v để debug
3. Đảm bảo đã cài đặt đủ dependencies
4. Verify token và license Hugging Face