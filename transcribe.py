import os
from dotenv import load_dotenv
import speech_recognition as sr
from pydub import AudioSegment
import torch
from pyannote.audio import Pipeline
from transformers import AutoModelForSequenceClassification
from pyannote.audio.core.model import Model
import wave
import contextlib
import datetime
import argparse
import sys
from pathlib import Path
import logging
import time
import torchaudio
from pydub.utils import mediainfo
import warnings
import threading
import json

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ProcessProgress:
    """Lớp theo dõi tiến trình xử lý"""
    def __init__(self, total_segments):
        self.current = 0
        self.total = total_segments
        self.start_time = time.time()
        self.status = "Đang khởi tạo..."
        self.lock = threading.Lock()

    def update(self, current, status=None):
        with self.lock:
            self.current = current
            if status:
                self.status = status

    def get_progress(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            progress = (self.current / self.total) * 100 if self.total > 0 else 0
            return {
                'current': self.current,
                'total': self.total,
                'progress': progress,
                'elapsed': elapsed,
                'status': self.status
            }

def display_progress(progress_tracker, refresh_rate=1):
    """Hiển thị tiến trình trong terminal"""
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        progress = progress_tracker.get_progress()
        
        elapsed = str(datetime.timedelta(seconds=int(progress['elapsed'])))
        progress_bar = '=' * int(progress['progress'] // 2) + '>' + ' ' * (50 - int(progress['progress'] // 2))
        
        print("\n=== TIẾN TRÌNH XỬ LÝ ===")
        print(f"[{progress_bar}] {progress['progress']:.1f}%")
        print(f"Đoạn: {progress['current']}/{progress['total']}")
        print(f"Thời gian: {elapsed}")
        print(f"Trạng thái: {progress['status']}")
        print("-" * 60)
        
        time.sleep(refresh_rate)

def suppress_warnings():
    """Ẩn các cảnh báo không cần thiết"""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="torchaudio._backend.soundfile_backend"
    )

def validate_token():
    """Kiểm tra token Hugging Face"""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        raise EnvironmentError(
            "\nKhông tìm thấy HUGGINGFACE_TOKEN!\n"
            "Vui lòng thực hiện các bước sau:\n"
            "1. Truy cập https://huggingface.co/pyannote/speaker-diarization và accept license\n"
            "2. Tạo token tại https://huggingface.co/settings/tokens\n"
            "3. Thiết lập token bằng một trong hai cách:\n"
            "   - Tạo file .env với nội dung: HUGGINGFACE_TOKEN=your_token\n"
            "   - Hoặc export HUGGINGFACE_TOKEN=your_token"
        )
    if not token.startswith('hf_'):
        raise ValueError(
            "\nToken không hợp lệ! Token phải bắt đầu bằng 'hf_'\n"
            "Vui lòng kiểm tra lại token tại https://huggingface.co/settings/tokens"
        )
    return token

def check_model_access():
    """Kiểm tra quyền truy cập vào các models cần thiết"""
    models = [
        "pyannote/speaker-diarization",
        "pyannote/segmentation"
    ]
    token = validate_token()
    
    for model_name in models:
        try:
            logger.info(f"Đang kiểm tra quyền truy cập vào {model_name}...")
            
            if "speaker-diarization" in model_name:
                Pipeline.from_pretrained(model_name, use_auth_token=token)
            else:
                Model.from_pretrained(model_name, use_auth_token=token)
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise Exception(
                    f"\nKhông có quyền truy cập vào model {model_name}!\n"
                    f"Vui lòng:\n"
                    f"1. Truy cập https://huggingface.co/{model_name}\n"
                    f"2. Đăng nhập và accept license\n"
                    f"3. Đảm bảo token có quyền read"
                )
            elif "404" in error_msg:
                raise Exception(f"\nKhông tìm thấy model {model_name}")
            else:
                raise Exception(f"Lỗi khi truy cập {model_name}: {error_msg}")

def setup_pipeline():
    """Khởi tạo pipeline với kiểm tra quyền truy cập"""
    token = validate_token()
    
    try:
        # Kiểm tra quyền truy cập trước
        check_model_access()
        
        # Khởi tạo pipeline
        logger.info("Đang khởi tạo pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=token
        )
        
        if pipeline is None:
            raise ValueError("Pipeline khởi tạo không thành công")
        
        return pipeline
        
    except Exception as e:
        error_msg = str(e)
        if "automatic-speech-recognition" in error_msg:
            raise Exception(
                "\nCần cài đặt thêm thư viện:\n"
                "pip install -U torch torchaudio\n"
                "pip install -U transformers"
            )
        else:
            raise Exception(f"Lỗi khi khởi tạo pipeline: {error_msg}")

def convert_timestamp(milliseconds):
    """Chuyển đổi milliseconds thành định dạng [HH:MM:SS]"""
    seconds = int(milliseconds / 1000)
    return str(datetime.timedelta(seconds=seconds))

def optimize_audio(audio_path, temp_dir, logger):
    """Tối ưu hóa audio trước khi xử lý"""
    try:
        logger.info("Đang tối ưu hóa audio...")
        
        # Lấy thông tin audio
        info = mediainfo(audio_path)
        sample_rate = int(info['sample_rate'])
        channels = int(info['channels'])
        
        # Tạo temporary path
        temp_wav = temp_dir / "temp.wav"
        optimized_wav = temp_dir / "optimized.wav"
        
        # Chuyển đổi MP3 sang WAV với parameters cụ thể
        audio = AudioSegment.from_mp3(audio_path)
        
        # Normalize audio
        normalized_audio = audio.normalize(headroom=0.1)
        
        # Convert to mono if stereo
        if channels > 1:
            logger.info("Chuyển đổi stereo sang mono...")
            normalized_audio = normalized_audio.set_channels(1)
        
        # Export với parameters tối ưu
        normalized_audio.export(
            str(temp_wav),
            format='wav',
            parameters=[
                '-ar', str(sample_rate),
                '-ac', '1',
                '-b:a', '192k'
            ]
        )
        
        # Đọc và resampling nếu cần
        waveform, sr = torchaudio.load(str(temp_wav))
        if sr != 16000:
            logger.info(f"Resampling từ {sr}Hz sang 16000Hz...")
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            torchaudio.save(str(optimized_wav), waveform, 16000)
        else:
            optimized_wav = temp_wav
        
        return optimized_wav
        
    except Exception as e:
        raise Exception(f"Lỗi khi tối ưu hóa audio: {str(e)}")

def prepare_audio(audio_path, temp_dir, logger):
    """Chuẩn bị file audio cho xử lý"""
    try:
        logger.info("Đang chuẩn bị file audio...")
        
        # Suppress warnings
        suppress_warnings()
        
        # Tạo thư mục temp nếu chưa tồn tại
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimize audio
        optimized_wav = optimize_audio(audio_path, temp_dir, logger)
        
        # Load optimized audio
        audio = AudioSegment.from_wav(str(optimized_wav))
        
        return audio, optimized_wav
        
    except Exception as e:
        raise Exception(f"Lỗi khi chuẩn bị file audio: {str(e)}")

def cleanup_temp_files(temp_dir, logger):
    """Dọn dẹp các file tạm"""
    try:
        for temp_file in temp_dir.glob("*"):
            temp_file.unlink()
        temp_dir.rmdir()
        logger.info("Đã dọn dẹp files tạm")
    except Exception as e:
        logger.warning(f"Lỗi khi dọn dẹp files tạm: {str(e)}")

def save_results(results, output_path, logger):
    """Lưu kết quả ra file và hiển thị"""
    output_content = ""
    for result in results:
        line = f"[{result['timestamp']}] {result['speaker']}: {result['text']}\n"
        output_content += line
        print(line, end='')
    
    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            logger.info(f"Đã lưu kết quả vào file: {output_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu file kết quả: {str(e)}")

def transcribe_segment(segment_audio, recognizer, retries=3):
    """Nhận dạng speech cho một đoạn audio với retry"""
    for attempt in range(retries):
        try:
            return recognizer.recognize_google(segment_audio, language="vi-VN")
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            if attempt == retries - 1:
                logger.error(f"Lỗi API Google Speech Recognition sau {retries} lần thử: {str(e)}")
                return None
            time.sleep(2)

def transcribe_audio(audio_path, output_path=None):
    """Chuyển đổi audio thành text với timestamp và người nói"""
    temp_dir = Path(f"temp_{Path(audio_path).stem}_{int(time.time())}")
    start_time = time.time()
    
    try:
        # Validate input file
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")
        
        file_extension = Path(audio_path).suffix.lower()
        if file_extension not in ['.mp3', '.wav', '.m4a', '.aac']:
            raise ValueError(f"Định dạng file {file_extension} không được hỗ trợ")
        
        logger.info(f"Bắt đầu xử lý file: {audio_path}")
        
        # Khởi tạo progress tracker
        progress_tracker = ProcessProgress(100)  # Số đoạn sẽ được cập nhật sau
        
        # Start progress display thread
        display_thread = threading.Thread(
            target=display_progress,
            args=(progress_tracker,),
            daemon=True
        )
        display_thread.start()
        
        # Convert audio nếu không phải MP3
        if file_extension != '.mp3':
            progress_tracker.update(0, f"Chuyển đổi {file_extension} sang MP3...")
            temp_mp3 = temp_dir / "temp.mp3"
            audio = AudioSegment.from_file(audio_path)
            audio.export(str(temp_mp3), format='mp3')
            audio_path = str(temp_mp3)
        
        # Chuẩn bị audio
        progress_tracker.update(0, "Đang chuẩn bị audio...")
        audio, temp_wav = prepare_audio(audio_path, temp_dir, logger)
        
        # Khởi tạo pipeline
        progress_tracker.update(0, "Đang khởi tạo pipeline...")
        pipeline = setup_pipeline()
        
        # Phân tích người nói
        progress_tracker.update(0, "Đang phân tích người nói...")
        diarization = pipeline(audio_path)
        
        # Đếm tổng số đoạn
        total_segments = len([_ for _ in diarization.itertracks(yield_label=True)])
        progress_tracker.total = total_segments
        
        # Khởi tạo recognizer
        recognizer = sr.Recognizer()
        
        results = []
        
        # Xử lý từng đoạn speech
        for idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True), 1):
            progress_tracker.update(
                idx,
                f"Đang xử lý đoạn {idx}/{total_segments}"
            )
            
            start_time_ms = turn.start * 1000
            end_time_ms = turn.end * 1000
            
            # Cắt đoạn audio
            segment = audio[start_time_ms:end_time_ms]
            temp_segment = temp_dir / "temp_segment.wav"
            segment.export(str(temp_segment), format="wav")
            
            # Nhận dạng speech
            with sr.AudioFile(str(temp_segment)) as source:
                segment_audio = recognizer.record(source)
                text = transcribe_segment(segment_audio, recognizer)
                
                if text:
                    results.append({
                        'timestamp': convert_timestamp(start_time_ms),
                        'speaker': speaker,
                        'text': text
                    })
        
        # Lưu kết quả
        progress_tracker.update(total_segments, "Đang lưu kết quả...")
        save_results(results, output_path, logger)
        
        # Thống kê
        duration = time.time() - start_time
        logger.info(f"Hoàn thành xử lý trong {duration:.2f} giây")
        logger.info(f"Số đoạn nhận dạng thành công: {len(results)}/{total_segments}")
        
        return results
        
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        raise
    finally:
        cleanup_temp_files(temp_dir, logger)

def main():
    """Hàm main xử lý tham số dòng lệnh"""
    parser = argparse.ArgumentParser(
        description='Chuyển đổi file audio thành văn bản có timestamp và phân biệt người nói'
    )
    parser.add_argument(
        'input',
        help='Đường dẫn đến file audio (hỗ trợ .mp3, .wav, .m4a, .aac)'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Đường dẫn file kết quả đầu ra'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Hiển thị thông tin chi tiết'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate input file
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Không tìm thấy file: {args.input}")
        
        # Create output directory if needed
        if args.output:
            output_dir = Path(args.output).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process file
        transcribe_audio(args.input, args.output)
        
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình.")
        sys.exit(0)
    except Exception as e:
        print(f"\nLỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()