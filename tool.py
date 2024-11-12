import os
from dotenv import load_dotenv
import speech_recognition as sr
from pydub import AudioSegment
import torch
from pyannote.audio import Pipeline
from transformers import AutoModelForSequenceClassification
from pyannote.audio.core.model import Model
import datetime
import argparse
import sys
from pathlib import Path
import logging
import time
import torchaudio
from pydub.utils import mediainfo
import warnings
import subprocess

# Thiết lập logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transcribe.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def suppress_warnings():
    """Ẩn tất cả các warning không cần thiết"""
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="torchaudio._backend.soundfile_backend"
    )
    # Ẩn warning từ mpg123
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*libmpg123.*"
    )
    # Tắt output từ mpg123
    os.environ['MPG123_QUIET'] = '1'
    # Thêm biến môi trường để ẩn warning từ ID3
    os.environ['SUPPRESS_ID3_WARNINGS'] = '1'

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
    logger.debug("Token validation successful")
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
            logger.info(f"Kiểm tra quyền truy cập: {model_name}")
            
            if "speaker-diarization" in model_name:
                Pipeline.from_pretrained(model_name, use_auth_token=token)
            else:
                Model.from_pretrained(model_name, use_auth_token=token)
            
            logger.info(f"Truy cập thành công: {model_name}")
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Lỗi khi truy cập {model_name}: {error_msg}")
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
    """Khởi tạo pipeline"""
    token = validate_token()
    logger.info("Bắt đầu khởi tạo pipeline...")
    
    try:
        check_model_access()
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=token
        )
        
        if pipeline is None:
            raise ValueError("Pipeline khởi tạo không thành công")
        
        logger.info("Khởi tạo pipeline thành công")
        return pipeline
        
    except Exception as e:
        logger.error(f"Lỗi khởi tạo pipeline: {str(e)}")
        raise

def convert_timestamp(milliseconds):
    """Chuyển đổi milliseconds thành [HH:MM:SS]"""
    seconds = int(milliseconds / 1000)
    return str(datetime.timedelta(seconds=seconds))

def clean_mp3(audio_path, temp_dir):
    """Làm sạch file MP3 trước khi xử lý"""
    logger.info("Đang làm sạch file MP3...")
    try:
        # Tạo file tạm
        temp_mp3 = temp_dir / "cleaned.mp3"
        
        # Sử dụng ffmpeg để tạo MP3 mới không có metadata và comment
        subprocess.run([
            'ffmpeg', '-i', str(audio_path),
            '-map_metadata', '-1',  # Xóa tất cả metadata
            '-id3v2_version', '0',  # Loại bỏ ID3 tags
            '-c:a', 'libmp3lame',  # Sử dụng MP3 encoder
            '-q:a', '0',  # Chất lượng cao nhất
            str(temp_mp3)
        ], check=True, capture_output=True, stderr=subprocess.PIPE)
        
        logger.info("Làm sạch file MP3 thành công")
        return str(temp_mp3)
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi làm sạch MP3: {e.stderr.decode()}")
        return audio_path
    except Exception as e:
        logger.error(f"Lỗi không xác định khi làm sạch MP3: {str(e)}")
        return audio_path

def optimize_audio(audio_path, temp_dir):
    """Tối ưu hóa audio"""
    logger.info("Bắt đầu tối ưu hóa audio...")
    
    try:
        # Lấy thông tin audio
        info = mediainfo(audio_path)
        sample_rate = int(info['sample_rate'])
        channels = int(info['channels'])
        
        logger.debug(f"Audio info: sample_rate={sample_rate}, channels={channels}")
        
        # Tạo temporary paths
        temp_wav = temp_dir / "temp.wav"
        optimized_wav = temp_dir / "optimized.wav"
        
        # Chuyển đổi sang WAV với warning suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = AudioSegment.from_mp3(audio_path)
        
        # Normalize audio
        logger.info("Đang normalize audio...")
        normalized_audio = audio.normalize(headroom=0.1)
        
        # Convert to mono if stereo
        if channels > 1:
            logger.info(f"Chuyển đổi {channels} channels sang mono...")
            normalized_audio = normalized_audio.set_channels(1)
        
        # Export với parameters tối ưu
        logger.info(f"Xuất file WAV tạm thời...")
        normalized_audio.export(
            str(temp_wav),
            format='wav',
            parameters=[
                '-ar', str(sample_rate),
                '-ac', '1',
                '-b:a', '192k'
            ]
        )
        
        # Resampling nếu cần
        waveform, sr = torchaudio.load(str(temp_wav))
        if sr != 16000:
            logger.info(f"Resampling từ {sr}Hz sang 16000Hz...")
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            torchaudio.save(str(optimized_wav), waveform, 16000)
            return optimized_wav
        
        logger.info("Tối ưu hóa audio hoàn tất")
        return temp_wav
        
    except Exception as e:
        logger.error(f"Lỗi khi tối ưu hóa audio: {str(e)}")
        raise

def prepare_audio(audio_path, temp_dir):
    """Chuẩn bị file audio với xử lý warning"""
    try:
        logger.info(f"Bắt đầu chuẩn bị file: {audio_path}")
        
        # Suppress warnings
        suppress_warnings()
        
        # Tạo thư mục temp
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean MP3 nếu là file MP3
        if audio_path.lower().endswith('.mp3'):
            audio_path = clean_mp3(audio_path, temp_dir)
        
        # Optimize audio với các warning đã được suppress
        optimized_wav = optimize_audio(audio_path, temp_dir)
        
        # Load optimized audio với các warning đã được suppress
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = AudioSegment.from_wav(str(optimized_wav))
        
        logger.info("Chuẩn bị audio hoàn tất")
        return audio, optimized_wav
        
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị audio: {str(e)}")
        raise

def cleanup_temp_files(temp_dir):
    """Dọn dẹp files tạm"""
    try:
        logger.info("Bắt đầu dọn dẹp files tạm...")
        for temp_file in temp_dir.glob("*"):
            temp_file.unlink()
        temp_dir.rmdir()
        logger.info("Dọn dẹp files tạm hoàn tất")
    except Exception as e:
        logger.warning(f"Lỗi khi dọn dẹp files tạm: {str(e)}")

def save_results(results, output_path):
    """Lưu và hiển thị kết quả"""
    output_content = ""
    for result in results:
        line = f"[{result['timestamp']}] {result['speaker']}: {result['text']}\n"
        output_content += line
        print(line, end='')
    
    if output_path:
        try:
            logger.info(f"Đang lưu kết quả vào: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            logger.info("Lưu kết quả thành công")
        except Exception as e:
            logger.error(f"Lỗi khi lưu file kết quả: {str(e)}")

def transcribe_segment(segment_audio, recognizer, retries=3):
    """Nhận dạng speech cho một đoạn"""
    for attempt in range(retries):
        try:
            text = recognizer.recognize_google(segment_audio, language="vi-VN")
            logger.debug(f"Nhận dạng thành công lần thử {attempt + 1}")
            return text
        except sr.UnknownValueError:
            logger.debug(f"Không nhận dạng được speech (lần {attempt + 1})")
            return None
        except sr.RequestError as e:
            logger.warning(f"Lỗi API (lần {attempt + 1}): {str(e)}")
            if attempt == retries - 1:
                logger.error(f"Lỗi API sau {retries} lần thử")
                return None
            time.sleep(2)

def transcribe_audio(audio_path, output_path=None):
    """Chuyển đổi audio thành text"""
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
        
        # Convert audio nếu không phải MP3
        if file_extension != '.mp3':
            logger.info(f"Chuyển đổi {file_extension} sang MP3...")
            temp_mp3 = temp_dir / "temp.mp3"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = AudioSegment.from_file(audio_path)
                audio.export(str(temp_mp3), format='mp3', parameters=[
                    '-map_metadata', '-1',  # Xóa metadata
                    '-id3v2_version', '0'   # Loại bỏ ID3 tags
                ])
            audio_path = str(temp_mp3)
        
        # Chuẩn bị audio
        audio, temp_wav = prepare_audio(audio_path, temp_dir)
        
        # Khởi tạo pipeline
        pipeline = setup_pipeline()
        
        # Phân tích người nói với warning suppressed
        logger.info("Đang phân tích người nói...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diarization = pipeline(audio_path)
        
        # Đếm số đoạn
        total_segments = len([_ for _ in diarization.itertracks(yield_label=True)])
        logger.info(f"Tổng số đoạn cần xử lý: {total_segments}")
        
        # Khởi tạo recognizer
        recognizer = sr.Recognizer()
        
        results = []
        processed = 0
        
        # Xử lý từng đoạn
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            processed += 1
            logger.info(f"Đang xử lý đoạn {processed}/{total_segments}")
            
            start_time_ms = turn.start * 1000
            end_time_ms = turn.end * 1000
            
            # Cắt đoạn audio
            logger.debug(f"Cắt đoạn {convert_timestamp(start_time_ms)} - {convert_timestamp(end_time_ms)}")
            segment = audio[start_time_ms:end_time_ms]
            temp_segment = temp_dir / "temp_segment.wav"
            segment.export(str(temp_segment), format="wav")
            
            # Nhận dạng speech với warning suppressed
            with sr.AudioFile(str(temp_segment)) as source:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    segment_audio = recognizer.record(source)
                    text = transcribe_segment(segment_audio, recognizer)
                
                if text:
                    results.append({
                        'timestamp': convert_timestamp(start_time_ms),
                        'speaker': speaker,
                        'text': text
                    })
                    logger.debug(f"Kết quả: [{convert_timestamp(start_time_ms)}] {speaker}: {text}")
        
        # Lưu kết quả
        save_results(results, output_path)
        
        # Thống kê
        duration = time.time() - start_time
        logger.info(f"Hoàn thành xử lý trong {duration:.2f} giây")
        logger.info(f"Số đoạn nhận dạng thành công: {len(results)}/{total_segments}")
        
        return results
        
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        raise
    finally:
        cleanup_temp_files(temp_dir)

def main():
    """Hàm main"""
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
        logger.setLevel(logging.DEBUG)
        # Thêm file handler cho debug
        debug_handler = logging.FileHandler('debug.log')
        debug_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)
    
    try:
        # Validate input file
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Không tìm thấy file: {args.input}")
        
        # Create output directory if needed
        if args.output:
            output_dir = Path(args.output).parent
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log thông tin file
        logger.info(f"File input: {args.input}")
        logger.info(f"File output: {args.output if args.output else 'Chỉ hiển thị kết quả'}")
        
        # Process file
        transcribe_audio(args.input, args.output)
        
    except KeyboardInterrupt:
        logger.warning("\nĐã dừng chương trình.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()