import os

# Force disable OneDNN and PIR API at the start to prevent internal engine crashes
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_mkldnn_enabled"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import sys
import logging
import datetime
import threading
import queue
import re
import io
import gc
import wave
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# Audio and Math Libraries
import torch
import sounddevice as sd
import numpy as np
from langdetect import detect, DetectorFactory

# OCR and Image Processing
import fitz  # PyMuPDF
from PIL import Image
from piper.voice import PiperVoice

# Document Support
import cv2
from docx import Document

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUTS_DIR = Path("outputs")
MODELS_DIR = Path("models")
INPUT_DIR = Path("input")
OUTPUTS_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)

# TTS Configuration
LANG_MODEL_MAP = {
    'en': 'en_US-lessac-medium.onnx',
    'hi': 'hi_IN-pratham-medium.onnx',
    'te': 'te_IN-maya-medium.onnx'
}

# OCR Configuration (PP-OCRv4 Server)
OCR_READER = None

# Local Server Model Paths (User should place models here for fully offline use)
SERVER_MODEL_PATHS = {
    'en': {
        'det': str(MODELS_DIR / "paddle/det_server"),
        'rec': str(MODELS_DIR / "paddle/rec_server_en"),
        'cls': str(MODELS_DIR / "paddle/cls_server")
    },
    'ml': {
        'det': str(MODELS_DIR / "paddle/det_server"),
        'rec': str(MODELS_DIR / "paddle/rec_server_ml"),
        'cls': str(MODELS_DIR / "paddle/cls_server")
    },
    'hi': { # Fallback to ml
        'det': str(MODELS_DIR / "paddle/det_server"),
        'rec': str(MODELS_DIR / "paddle/rec_server_ml"),
        'cls': str(MODELS_DIR / "paddle/cls_server")
    },
    'te': { # Fallback to ml
        'det': str(MODELS_DIR / "paddle/det_server"),
        'rec': str(MODELS_DIR / "paddle/rec_server_ml"),
        'cls': str(MODELS_DIR / "paddle/cls_server")
    }
}

def detect_document_script(img):
    """Fast pass to detect dominant script (te/hi/en) before full OCR."""
    from paddleocr import PaddleOCR
    logger.info("Detecting document script...")
    
    try:
        # Use 'en' for detection pass (it's sufficient to find boxes)
        temp_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        res = temp_reader.ocr(img, cls=True)
        
        if not res or not res[0]:
            return 'en'
            
        # Sample text to check scripts
        text_sample = " ".join([line[1][0] for line in res[0][:15]]) # First 15 lines for better sampling
        detected = detect_language_robust(text_sample)
        logger.info(f"Script Detected: {detected}")
        return detected
    except Exception as e:
        logger.warning(f"Script detection failed: {e}. Falling back to 'en'.")
        return 'en'

def adaptive_resize(img, target_long_side=3000):
    """Resizes image so its longest side matches target_long_side while maintaining aspect ratio."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > target_long_side or long_side < 1500: # Only resize if too big or too small
        scale = target_long_side / long_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img

def perform_smart_ocr(img, detected_lang=None):
    """OCR handler that reconstructs sentences from detected blocks."""
    if not detected_lang:
        detected_lang = detect_document_script(img)
    
    reader = get_ocr_reader(detected_lang)
    res = reader.ocr(img, cls=True)
    
    if not res or not res[0]:
        return ""

    # res[0] contains list of [[box], [text, confidence]]
    # Sort blocks primarily by Y-coordinate (top to bottom)
    # This helps in grouping words into the same visual line
    data = res[0]
    data.sort(key=lambda x: x[0][0][1])

    lines = []
    if data:
        current_line = [data[0]]
        for i in range(1, len(data)):
            # Calculate dynamic threshold based on the height of the previous word
            prev_box = current_line[-1][0]
            prev_height = max(10, prev_box[3][1] - prev_box[0][1])
            y_threshold = prev_height * 0.6  # 60% of the word's height
            
            # Compare current word's Y with the Y of the previous word on the current line
            current_y = data[i][0][0][1]
            prev_y = current_line[-1][0][0][1]
            
            if abs(current_y - prev_y) < y_threshold:
                current_line.append(data[i])
            else:
                # Sort the completed line horizontally (left to right)
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(" ".join([word[1][0] for word in current_line]))
                current_line = [data[i]]
        
        # Add the last line
        current_line.sort(key=lambda x: x[0][0][0])
        lines.append(" ".join([word[1][0] for word in current_line]))
    
    return "\n".join(lines)


def get_ocr_reader(lang='te'):
    """Initialize PaddleOCR with high-accuracy server models and aggressive tuning."""
    global OCR_READER
    
    # Handle model switching and memory cleanup
    if OCR_READER is not None:
        if getattr(OCR_READER, '_lang', None) != lang:
            logger.info(f"Purging old model ({getattr(OCR_READER, '_lang', None)}) to free memory...")
            del OCR_READER
            OCR_READER = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            return OCR_READER

    from paddleocr import PaddleOCR
    # For Indian languages, use 'ml' (multilingual) to handle English code-mixing
    actual_lang = 'ml' if lang in ['hi', 'te'] else lang
    logger.info(f"Initializing High-Accuracy OCR (lang={actual_lang}, original={lang})...")
    
    # Get local model paths if they exist, otherwise download defaults
    paths = SERVER_MODEL_PATHS.get(lang, SERVER_MODEL_PATHS['en'])
    
    # Helper to check if a directory contains a valid PaddleOCR model
    def is_valid_model(p):
        return os.path.isdir(p) and any(f.endswith('.pdmodel') for f in os.listdir(p))

    kw = {
        "use_angle_cls": True,
        "lang": actual_lang,          # Use detected language!
        "rec_algorithm": 'SVTR_LCNet', # Specialized handwriting recognition algorithm
        "show_log": False,
        "use_gpu": torch.cuda.is_available(),
        "download_enabled": True,   
        
        # High-Accuracy Settings
        "det_limit_side_len": 2560,   # Prevent extreme downscaling for high-res scans
        "det_db_unclip_ratio": 1.6,   # Stable for both printed and handwritten
        "det_db_thresh": 0.3,         # Filter out faint noise
        "det_db_box_thresh": 0.5,     # Require stronger confidence for text blocks
        "rec_batch_num": 1,
        
        # Use server models ONLY if they are already downloaded and valid
        "det_model_dir": paths['det'] if is_valid_model(paths['det']) else None,
        "rec_model_dir": paths['rec'] if is_valid_model(paths['rec']) else None,
        "cls_model_dir": paths['cls'] if is_valid_model(paths['cls']) else None
    }

    OCR_READER = PaddleOCR(**kw)
    OCR_READER._lang = lang
    return OCR_READER

def deskew_image(img):
    """Safely corrects image tilt without introducing distortions."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect edges/text for orientation
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Filter out noise/borders by looking at a central crop
    h, w = thresh.shape
    margin_h, margin_w = h // 10, w // 10
    crop = thresh[margin_h:h-margin_h, margin_w:w-margin_w]
    
    coords = np.column_stack(np.where(crop > 0))
    if len(coords) < 10: return img # Not enough text to deskew
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Handle OpenCV angle variations (0-90 vs -90-0)
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    
    # Ignore tiny or massive rotations (sanity check)
    if abs(angle) < 0.1 or abs(angle) > 15:
        return img
        
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    logger.info(f"Deskewing: Corrected {angle:.2f} degree tilt.")
    return rotated

def enhance_image(img_bgr):
    """Advanced Enhancement: CLAHE contrast and Unsharp Masking for crisp text."""
    # 1. Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Contrast Enhancement (CLAHE)
    # This makes text "pop" regardless of lighting conditions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Unsharp Masking for Sharpening
    # Bilateral filter can sometimes smudge thin handwriting strokes. 
    # Unsharp masking sharpens edges without destroying thin strokes.
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    
    # Return as BGR (PaddleOCR expects 3 channels)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

def review_and_edit_ui(text):
    """The Safety Net: Human-in-the-Loop review and correction window."""
    try:
        root = tk.Tk()
        root.withdraw() # Hide main window
        
        # Simple text editor window
        editor = tk.Toplevel()
        editor.title("OCR Proofreader - Review & Edit")
        editor.geometry("800x600")
        
        txt_widget = tk.Text(editor, font=("Consolas", 12), undo=True)
        txt_widget.insert("1.0", text)
        txt_widget.pack(expand=True, fill="both", padx=10, pady=10)
        
        edited_text = [text] # Use list to store result from closure
        
        def on_save():
            edited_text[0] = txt_widget.get("1.0", "end-1c")
            editor.destroy()
            root.quit()
            
        btn = tk.Button(editor, text="Save & Continue to Speech", command=on_save, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"))
        btn.pack(pady=10)
        
        root.mainloop()
        return edited_text[0]
    except Exception as e:
        logger.warning(f"Could not open Review UI (likely no display): {e}")
        return text

def save_extracted_text(text):
    """Save text only for scanned documents."""
    if not text: return
    try:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        (OUTPUTS_DIR / f"ocr_extracted_{ts}.txt").write_text(text, encoding="utf-8")
        (OUTPUTS_DIR / "latest_extraction.txt").write_text(text, encoding="utf-8")
        logger.info(f"Scanned text saved to outputs folder.")
    except Exception as e:
        logger.warning(f"Could not save extraction: {e}")

def extract_text_from_scanned_pdf(file_path):
    """Extraction Process for Scanned PDFs (Generator)."""
    doc = fitz.open(file_path)
    detected_lang = None
    
    for i, page in enumerate(doc):
        logger.info(f"Processing page {i+1}/{len(doc)}...")
        # 4.0x zoom (Restoring high-coverage resolution)
        zoom = 4.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        
        # Convert pixmap to BGR image for OpenCV/OCR
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # DISABLED auto-resizing to preserve detail for handwriting OCR
        # img = resize_image_if_needed(img)
        img = deskew_image(img) # Correct orientation before processing
        img = enhance_image(img)
        
        # Detect script only once on the first page
        if i == 0:
            detected_lang = detect_document_script(img)
        
        # Perform single-pass OCR
        page_text = perform_smart_ocr(img, detected_lang=detected_lang)
        if page_text:
            yield page_text
            
    doc.close()

# --- AUDITORY CORE (Dynamic Detection & Streaming TTS) ---

def detect_language_robust(text):
    """Detect language using Unicode ranges for English, Hindi, and Telugu."""
    if not text: return 'en'
    # Telugu: \u0C00-\u0C7F
    telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
    # Hindi/Devanagari: \u0900-\u097F
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    
    total_alpha = len(re.findall(r'[a-zA-Z\u0C00-\u0C7F\u0900-\u097F]', text))
    if total_alpha == 0: return 'en'
    
    # Extremely sensitive threshold for scripts (even 1 character triggers it)
    if telugu_chars >= 1: return 'te'
    if hindi_chars >= 1: return 'hi'
    
    return 'en'

def split_into_sentences(text):
    """Split text into chunks for sentence-by-sentence streaming."""
    chunks = re.split(r'(?<=[.!?।])\s+|\n+', text)
    return [c.strip() for c in chunks if c.strip()]

def run_tts(text):
    """Streaming TTS with per-sentence language detection."""
    sentences = split_into_sentences(text)
    if not sentences: return
    
    logger.info(f"Streaming {len(sentences)} sentences...")
    audio_queue = queue.Queue(maxsize=3)
    loaded_voices = {}

    def synthesizer():
        try:
            for s in sentences:
                lang = detect_language_robust(s)
                model_name = LANG_MODEL_MAP.get(lang, LANG_MODEL_MAP['en'])
                model_path = MODELS_DIR / model_name
                
                if model_name not in loaded_voices:
                    logger.info(f"Loading voice: {model_name}")
                    loaded_voices[model_name] = PiperVoice.load(model_path)
                
                voice = loaded_voices[model_name]
                
                # Collect all audio chunks from the synthesis generator
                audio_data = b""
                for chunk in voice.synthesize(s):
                    if isinstance(chunk, bytes):
                        audio_data += chunk
                    else:
                        # Handle AudioChunk objects in newer Piper versions
                        audio_data += chunk.audio_int16_bytes
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                if len(audio_np) > 0:
                    audio_queue.put((audio_np, voice.config.sample_rate, lang))
                else:
                    logger.warning(f"Synthesis returned empty audio for sentence: {s[:30]}...")
            audio_queue.put(None)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            audio_queue.put(None)

    threading.Thread(target=synthesizer, daemon=True).start()

    all_audio_chunks = []
    final_rate = 22050
    
    while True:
        item = audio_queue.get()
        if item is None: break
        audio_np, sample_rate, lang = item
        all_audio_chunks.append(audio_np)
        final_rate = sample_rate
        
        logger.info(f"Playing ({lang}): {len(audio_np)/sample_rate:.1f}s")
        sd.play(audio_np, sample_rate)
        sd.wait()

    # Save the full audio to the outputs folder
    if all_audio_chunks:
        try:
            combined_audio = np.concatenate(all_audio_chunks)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = OUTPUTS_DIR / f"speech_output_{ts}.wav"
            
            with wave.open(str(audio_file), "wb") as wav_file:
                wav_file.setnchannels(1) # Mono
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(final_rate)
                wav_file.writeframes(combined_audio.astype(np.int16).tobytes())
                
            logger.info(f"Full audio saved to: {audio_file}")
        except Exception as e:
            logger.warning(f"Could not save audio file: {e}")

# --- MAIN WORKFLOW ---

def main():
    print("\n--- Unified Multilingual OCR & Streaming TTS ---")
    
    # 1. Ask user to select file
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not file_path or not os.path.exists(file_path):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Document", 
            filetypes=[
                ("Supported Files", "*.pdf *.jpg *.jpeg *.png *.webp *.txt *.docx"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        
    if not file_path: return

    ext = Path(file_path).suffix.lower()
    text = ""
    needs_review = False
    
    # 2. Process based on type
    if ext == ".pdf":
        doc = fitz.open(file_path)
        digital_text = "".join([page.get_text() for page in doc]).strip()
        if len(digital_text) > 150:
            logger.info("Direct Workflow: Digital PDF detected.")
            text = digital_text
            save_extracted_text(text) # Now saves digital PDFs too
        else:
            needs_review = True
            # Process as generator to allow progress tracking/future unblocking
            extracted_pages = []
            for page_text in extract_text_from_scanned_pdf(file_path):
                extracted_pages.append(page_text)
            
            text = "\n".join(extracted_pages)
            if text:
                # Always save a quick draft for reference
                draft_path = OUTPUTS_DIR / "latest_draft.txt"
                draft_path.write_text(text, encoding='utf-8')
                logger.info(f"Text extraction complete. Saved to {draft_path}")
                save_extracted_text(text)
    
    elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
        logger.info(f"Image detected. Processing...")
        needs_review = True
        raw_img = cv2.imread(file_path)
        
        # Adaptive resize and light enhancement
        img = adaptive_resize(raw_img)
        img = deskew_image(img) # Correct orientation before processing
        img = enhance_image(img)
        
        # Perform script detection and single-pass OCR
        text = perform_smart_ocr(img)
        
        if text:
            # Always save a quick draft for reference
            draft_path = OUTPUTS_DIR / "latest_draft.txt"
            draft_path.write_text(text, encoding='utf-8')
            logger.info(f"Text extraction complete. Saved to {draft_path}")
            save_extracted_text(text)
        
    elif ext in [".txt", ".docx"]:
        logger.info("Direct Workflow: Normal document detected.")
        if ext == ".txt":
            text = Path(file_path).read_text(errors='ignore')
        else:
            text = "\n".join([p.text for p in Document(file_path).paragraphs])
        
        if text:
            save_extracted_text(text) # Now saves Word and Text docs too

    # 3. Read aloud using streaming TTS (with Human-in-the-Loop for scanned docs)
    if text:
        if needs_review:
            # Prompt for review before starting speech
            text = review_and_edit_ui(text)
        run_tts(text)
    else:
        logger.warning("No text extracted.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nSession ended.")
    except Exception as e:
        logger.error(f"Error: {e}")