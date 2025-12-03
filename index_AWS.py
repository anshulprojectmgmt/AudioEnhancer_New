# =========================================================================================================
#                    IMPORTS & INITIAL SETUP
# =========================================================================================================
import time, logging, shutil, subprocess, os, json, uuid, warnings, boto3, pandas, requests, re, asyncio, psutil
import aiohttp # <--- Added for async HTTP requests
from itertools import chain
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import google.generativeai as genai
# import torch # <-- Removed torch (unless needed elsewhere)
# import multiprocessing # <-- Removed multiprocessing
# import concurrent.futures # <-- Removed concurrent.futures
# from voice_clone.src.chatterbox.tts import tts_generate_segment , get_tts_model # <-- Removed local TTS imports
import base64 # <-- Added for encoding audio

# --- Load Environment Variables ---
load_dotenv()

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", module="whisper")
warnings.filterwarnings("ignore", message=".*bytes read.*")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Performance Monitoring Setup ---
PROCESS = psutil.Process(os.getpid())
INITIAL_MEMORY = PROCESS.memory_info().rss / (1024 * 1024)
logger.info(f"Initial RAM Usage: {INITIAL_MEMORY:.2f} MB")

# =========================================================================================================
#                    RUNPOD CONFIGURATION <--- NEW SECTION
# =========================================================================================================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY") # Make sure this is in your .env file
RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL") # Make sure this is in your .env file (the /runsync URL)
MAX_CONCURRENT_RUNPOD_REQUESTS = 5 # Control how many requests hit RunPod simultaneously (adjust as needed for cost/speed)

if not RUNPOD_API_KEY:
    logger.error("RUNPOD_API_KEY not found in environment variables.")
    # You might want to raise an exception or exit here in a real app
if not RUNPOD_ENDPOINT_URL:
    logger.error("RUNPOD_ENDPOINT_URL not found in environment variables.")
    # You might want to raise an exception or exit here in a real app

# =========================================================================================================
#                    ENHANCED PERFORMANCE MONITORING
# =========================================================================================================
# (log_performance function remains the same)
def log_performance(step_name, start_time):
    end_time = time.time()
    duration = end_time - start_time
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    swap_usage = psutil.swap_memory().percent
    print("\n" + "="*80)
    print(f"--- PERFORMANCE REPORT: [{step_name}] ---")
    print(f"    - Duration: {duration:.2f} seconds")
    print(f"    - System CPU Load: {cpu_usage}%")
    print(f"    - System RAM Usage: {ram_usage}%")
    print(f"    - System SWAP Usage: {swap_usage}%")
    print("="*80 + "\n")


# --- Initialize FastAPI App ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Service Clients Setup (Google, AWS, Gemini) ---
# (Remains the same)
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
sa_info = json.loads(os.getenv("GOOGLE_SA_JSON"))
credentials = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
sheets_service = build('sheets', 'v4', credentials=credentials, cache_discovery=False)
drive_service = build('drive', 'v3', credentials=credentials)
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
S3_BUCKET = os.getenv("S3_BUCKET")

# =========================================================================================================
#                    HELPER FUNCTIONS
# =========================================================================================================

# (run_ffmpeg, upload_file, download_file, call_gemini remain the same)
def run_ffmpeg(cmd, desc):
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"[FFmpeg] {desc} completed successfully.")
            return
        except subprocess.CalledProcessError as e:
            attempt += 1
            err = e.stderr if e.stderr else e.stdout
            logger.error(f"[FFmpeg] {desc} failed (attempt {attempt}/{MAX_RETRIES}): {err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"[FFmpeg] {desc} failed permanently: {err}")

def upload_file(local_path: str, s3_key: str):
    try:
        s3.upload_file(Filename=local_path, Bucket=S3_BUCKET, Key=s3_key)
        s3_url = f"https://{S3_BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
        logger.info(f"Successfully uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
        return {"status": "success", "url": s3_url, "s3_key": s3_key}
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to S3: {e}")
        raise RuntimeError(f"S3 upload failed for {local_path}")

def download_file(s3_key: str, local_path: str):
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(Bucket=S3_BUCKET, Key=s3_key, Filename=local_path)
        logger.info(f"Successfully downloaded s3://{S3_BUCKET}/{s3_key} to {local_path}")
        return {"status": "success", "file_path": local_path}
    except Exception as e:
        logger.error(f"Failed to download {s3_key} from S3: {e}")
        raise RuntimeError(f"S3 download failed for {s3_key}")

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    return model.generate_content(prompt).text.strip()

# def tts_worker_init(...): # <-- Removed

# (get_media_duration, create_word_highlighted_ass remain the same)
def get_media_duration(path: str) -> float:
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path
        ], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        logger.warning(f"Could not get duration for {path}. Defaulting to 0.0")
        return 0.0

def create_word_highlighted_ass(words, ass_path, words_per_caption=7):
    # (Function content remains the same)
    def ass_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 100)
        return f"{h:d}:{m:02d}:{s:02d}.{ms:02d}"

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,75,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
Style: Highlight,Arial,75,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for i in range(0, len(words), words_per_caption):
        word_group = words[i:i + words_per_caption]
        for j, word in enumerate(word_group):
            start = ass_time(word["start"])
            end = ass_time(word["end"])
            line = []
            for k, w in enumerate(word_group):
                text = w["punctuated_word"]
                if k == j:
                    line.append(r"{\rHighlight}" + text + r"{\rDefault}")
                else:
                    line.append(text)
            text = " ".join(line).strip()
            dialogue = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            events.append(dialogue)

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(header + "\n".join(events))

# (process_segments_with_ffmpeg remains the same - it uses the generated audio paths)
def process_segments_with_ffmpeg(segments, input_path, output_path, ass_path, tmp_base):
    # (Function content remains the same)
    t0 = time.time()
    tmp_audio = os.path.join(tmp_base, "audio.wav")
    
    # === STEP 1: Build audio track ===
    audio_inputs = [seg.get("audio_path") for seg in segments if seg.get("audio_path") and os.path.exists(seg.get("audio_path"))]
    unique_audio_inputs = sorted(list(set(audio_inputs)))
    audio_filters = []
    concat_inputs = []
    for i, seg in enumerate(segments):
        duration = (seg["end"] - seg["start"]) / seg["factor"]
        audio_path = seg.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            aud_idx = unique_audio_inputs.index(audio_path)
            audio_filters.append(f"[{aud_idx}:a]atrim=0:{duration},aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=PTS-STARTPTS[a{i}]")
        else: # Silent audio for segments without an audio path
            audio_filters.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{duration},asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs.append(f"[a{i}]")

    if concat_inputs:
        concat_filter = f"{''.join(concat_inputs)}concat=n={len(segments)}:v=0:a=1[outa]"
        filter_complex = ";".join(audio_filters + [concat_filter])
        audio_cmd = ["ffmpeg", "-y", *chain.from_iterable([["-i", p] for p in unique_audio_inputs]), "-filter_complex", filter_complex, "-map", "[outa]", "-c:a", "pcm_s16le", "-ar", "48000", tmp_audio]
        run_ffmpeg(audio_cmd, "Audio Track Processing")
        log_performance("FFmpeg - Audio Stitching", t0)
    else: # If no audio inputs, create a silent track for the whole duration
        total_duration = sum([(s["end"] - s["start"]) / s["factor"] for s in segments])
        run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo:d={total_duration}", "-c:a", "pcm_s16le", tmp_audio], "Silent Audio Generation")
        log_performance("FFmpeg - Silent Audio Generation", t0)

    # === STEP 2: Transcribe for Captions ===
    t1 = time.time()
    with open(tmp_audio, "rb") as audio_file:
        payload: FileSource = {"buffer": audio_file.read()}
    options = PrerecordedOptions(model="nova-2", smart_format=True)
    # Ensure DEEPGRAM_API_KEY is loaded via load_dotenv()
    deepgram_client = DeepgramClient() 
    response = deepgram_client.listen.rest.v("1").transcribe_file(payload, options, timeout=300)
    dg_data = json.loads(response.to_json())
    log_performance("Deepgram - Caption Generation", t1)
    header = """[Script Info]
    ScriptType: v4.00+
    PlayResX: 1920
    PlayResY: 1080
    [V4+ Styles]
    Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    Style: Default,DejaVu Sans,75,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
    Style: Highlight,Arial,75,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1
    [Events]
    Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    """
    
    words = []
    if dg_data.get("results") and dg_data["results"].get("channels") and dg_data["results"]["channels"][0].get("alternatives"):
        words = dg_data["results"]["channels"][0]["alternatives"][0].get("words", [])
        
    if words:
         create_word_highlighted_ass(words, ass_path)
    else:
        logger.warning("No words found in Deepgram transcription for captions.")
        # Create an empty ass file or handle as needed
        with open(ass_path, 'w') as f:
            f.write(header) # Write header only

    # === STEP 3: Create final video ===
    t2 = time.time()
    video_inputs = sorted(list(set([seg["path"] for seg in segments])))
    video_filters = []
    video_concat_inputs = []
    for i, seg in enumerate(segments):
        vid_idx = video_inputs.index(seg["path"])
        speed_filter = f",setpts=(PTS-STARTPTS)/{seg['factor']}" if seg['factor'] != 1.0 else ",setpts=PTS-STARTPTS"
        video_filters.append(f"[{vid_idx}:v]trim=start={seg['start']}:end={seg['end']}{speed_filter},scale=1920:1080,setsar=1[v{i}]")
        video_concat_inputs.append(f"[v{i}]")
    
    video_concat_filter = f"{''.join(video_concat_inputs)}concat=n={len(segments)}:v=1:a=0[outv]"
    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    
    # Check if ass file exists and has content before adding the filter
    ass_filter_str = f";[outv]ass='{escaped_ass}'[vout]" if os.path.exists(ass_path) and os.path.getsize(ass_path) > len(header) else "[outv]"
    map_video_output = "[vout]" if ass_filter_str != "[outv]" else "[outv]" # Use [vout] if filter applied, else [outv]
    
    final_filter = f"{';'.join(video_filters + [video_concat_filter])}{ass_filter_str if map_video_output == '[vout]' else ''}"

    final_cmd = ["ffmpeg", "-y", *chain.from_iterable([["-i", p] for p in video_inputs]), "-i", tmp_audio, "-filter_complex", final_filter, "-map", map_video_output, "-map", f"{len(video_inputs)}:a", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", output_path]
    run_ffmpeg(final_cmd, "Final Video Composition")
    log_performance("FFmpeg - Final Video Composition", t2)
    return output_path, tmp_audio

# =========================================================================================================
#                    NEW RUNPOD TTS HELPER FUNCTION <--- NEW SECTION
# =========================================================================================================
# async def call_runpod_tts(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, text: str, ref_audio_path: str, output_path: str, segment_index: int):
#     """
#     Sends a TTS request to the RunPod endpoint, handles response, and saves audio.
#     Includes semaphore for concurrency control and retries.
#     """
#     MAX_RETRIES = 3
#     RETRY_DELAY = 5 # seconds

#     # 1. Encode reference audio
#     try:
#         with open(ref_audio_path, 'rb') as audio_file:
#             audio_bytes = audio_file.read()
#             ref_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
#     except FileNotFoundError:
#         logger.error(f"[RunPod TTS {segment_index}] Reference audio not found: {ref_audio_path}")
#         return False # Indicate failure
#     except Exception as e:
#         logger.error(f"[RunPod TTS {segment_index}] Error encoding ref audio {ref_audio_path}: {e}")
#         return False

#     # 2. Prepare payload
#     payload = {
#         "input": {
#             "task": "tts",
#             "text": text,
#             "ref_audio_b64": ref_audio_b64,
#             # Add other optional parameters if needed (e.g., exaggeration)
#             # "exaggeration": 0.5
#         }
#     }
#     headers = {
#         "Authorization": f"Bearer {RUNPOD_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     # 3. Make request with semaphore and retries
#     async with semaphore:
#         logger.info(f"[RunPod TTS {segment_index}] Sending request...")
#         for attempt in range(MAX_RETRIES):
#             try:
#                 async with session.post(RUNPOD_ENDPOINT_URL, json=payload, headers=headers, timeout=300) as response: # 5 min timeout
#                     response_text = await response.text() # Read text first for debugging
#                     if response.status == 200:
#                         response_data = json.loads(response_text)
#                         status = response_data.get("status")

#                         if status == "COMPLETED":
#                             output = response_data.get("output", {})
#                             if output.get("audio_b64"):
#                                 # Decode and save audio
#                                 audio_bytes_out = base64.b64decode(output["audio_b64"])
#                                 with open(output_path, "wb") as f_out:
#                                     f_out.write(audio_bytes_out)
#                                 logger.info(f"[RunPod TTS {segment_index}] Successfully generated: {output_path}")
#                                 return True # Indicate success
#                             else:
#                                 error_msg = output.get("error", "Unknown error from worker")
#                                 logger.error(f"[RunPod TTS {segment_index}] Worker failed: {error_msg}")
#                                 return False # Indicate failure (don't retry worker errors)
#                         elif status == "FAILED":
#                              error_details = response_data.get("error", "No error details provided.")
#                              logger.error(f"[RunPod TTS {segment_index}] Job FAILED on RunPod: {error_details}")
#                              return False # Indicate failure (don't retry job fails)
#                         else:
#                              # Handle IN_QUEUE, IN_PROGRESS (shouldn't happen with /runsync but handle defensively)
#                              logger.warning(f"[RunPod TTS {segment_index}] Unexpected status {status}. Retrying...")
                             
#                     else:
#                         logger.error(f"[RunPod TTS {segment_index}] Request failed (Attempt {attempt+1}/{MAX_RETRIES}): Status {response.status}, Body: {response_text[:500]}") # Log truncated body

#             except asyncio.TimeoutError:
#                 logger.error(f"[RunPod TTS {segment_index}] Request timed out (Attempt {attempt+1}/{MAX_RETRIES}).")
#             except aiohttp.ClientError as e:
#                  logger.error(f"[RunPod TTS {segment_index}] Client error (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
#             except Exception as e:
#                 logger.error(f"[RunPod TTS {segment_index}] Unexpected error during request (Attempt {attempt+1}/{MAX_RETRIES}): {e}")

#             # Wait before retrying
#             if attempt < MAX_RETRIES - 1:
#                 await asyncio.sleep(RETRY_DELAY * (attempt + 1)) # Exponential backoff might be better
#             else:
#                  logger.error(f"[RunPod TTS {segment_index}] Max retries reached. Failed to generate.")
#                  return False # Indicate final failure

#     return False # Should not be reached if semaphore logic is correct

async def call_runpod_tts(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, text: str, ref_audio_b64: str, output_path: str, segment_index: int):
    """
    Sends a TTS request to the RunPod endpoint.
    Accepts pre-encoded Base64 audio string to save memory.
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 5 # seconds

    # --- CHANGE: File reading/encoding logic REMOVED from here ---
    
    # 2. Prepare payload (Using the passed b64 string directly)
    payload = {
        "input": {
            "task": "tts",
            "text": text,
            "ref_audio_b64": ref_audio_b64, # <--- Uses the argument passed in
            # Add other optional parameters if needed (e.g., exaggeration)
            # "exaggeration": 0.5
        }
    }
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    # 3. Make request with semaphore and retries
    async with semaphore:
        logger.info(f"[RunPod TTS {segment_index}] Sending request...")
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(RUNPOD_ENDPOINT_URL, json=payload, headers=headers, timeout=300) as response: # 5 min timeout
                    response_text = await response.text() # Read text first for debugging
                    if response.status == 200:
                        response_data = json.loads(response_text)
                        status = response_data.get("status")

                        if status == "COMPLETED":
                            output = response_data.get("output", {})
                            if output.get("audio_b64"):
                                # Decode and save audio
                                audio_bytes_out = base64.b64decode(output["audio_b64"])
                                with open(output_path, "wb") as f_out:
                                    f_out.write(audio_bytes_out)
                                logger.info(f"[RunPod TTS {segment_index}] Successfully generated: {output_path}")
                                return True # Indicate success
                            else:
                                error_msg = output.get("error", "Unknown error from worker")
                                logger.error(f"[RunPod TTS {segment_index}] Worker failed: {error_msg}")
                                return False # Indicate failure (don't retry worker errors)
                        elif status == "FAILED":
                             error_details = response_data.get("error", "No error details provided.")
                             logger.error(f"[RunPod TTS {segment_index}] Job FAILED on RunPod: {error_details}")
                             return False # Indicate failure (don't retry job fails)
                        else:
                             # Handle IN_QUEUE, IN_PROGRESS
                             logger.warning(f"[RunPod TTS {segment_index}] Unexpected status {status}. Retrying...")
                             
                    else:
                        logger.error(f"[RunPod TTS {segment_index}] Request failed (Attempt {attempt+1}/{MAX_RETRIES}): Status {response.status}, Body: {response_text[:500]}")

            except asyncio.TimeoutError:
                logger.error(f"[RunPod TTS {segment_index}] Request timed out (Attempt {attempt+1}/{MAX_RETRIES}).")
            except aiohttp.ClientError as e:
                 logger.error(f"[RunPod TTS {segment_index}] Client error (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            except Exception as e:
                logger.error(f"[RunPod TTS {segment_index}] Unexpected error during request (Attempt {attempt+1}/{MAX_RETRIES}): {e}")

            # Wait before retrying
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                 logger.error(f"[RunPod TTS {segment_index}] Max retries reached. Failed to generate.")
                 return False # Indicate final failure

    return False

# =========================================================================================================
#                    ENDPOINT 1: /process-video
# =========================================================================================================
# (Remains the same - no TTS here)
@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    # (Function content remains the same)
    t_start = time.time()
    uid = str(uuid.uuid4())
    filename = f"{os.path.splitext(file.filename)[0]}_{uid}.mp4"
    
    local_dir = f"./Data/tmp/{uid}"
    os.makedirs(local_dir, exist_ok=True)
    local_video_path = os.path.join(local_dir, filename)
    local_audio_path = os.path.join(local_dir, "extracted_audio.wav")

    try:
        t0 = time.time()
        with open(local_video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        log_performance("Save Video", t0)

        t1 = time.time()
        run_ffmpeg(["ffmpeg", "-y", "-i", local_video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", local_audio_path], "Audio Extraction")
        log_performance("Extract Audio", t1)
        
        t2 = time.time()
        with open(local_audio_path, "rb") as audio_file:
            payload: FileSource = {"buffer": audio_file.read()}
        # Ensure DEEPGRAM_API_KEY is loaded via load_dotenv()
        deepgram_client = DeepgramClient()
        options = PrerecordedOptions(model="nova-3", smart_format=True, diarize=True)
        response = deepgram_client.listen.rest.v("1").transcribe_file(payload, options, timeout=300)
        dg_data = json.loads(response.to_json())
        log_performance("Deepgram Transcription", t2)
        
        paragraphs = []
        if dg_data.get("results") and dg_data["results"].get("channels") and dg_data["results"]["channels"][0].get("alternatives"):
             paragraphs = dg_data["results"]["channels"][0]["alternatives"][0].get("paragraphs", {}).get("paragraphs", [])

        segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for p in paragraphs for s in p["sentences"]]
        
        words = []
        if dg_data.get("results") and dg_data["results"].get("channels") and dg_data["results"]["channels"][0].get("alternatives"):
            words = dg_data["results"]["channels"][0]["alternatives"][0].get("words", [])

        speaker_durations = {}
        word_speakers = []
        for w in words:
            speaker = w.get("speaker")
            word_speakers.append({"start": w["start"], "end": w["end"], "speaker": speaker})
            if speaker is not None:
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + (w["end"] - w["start"])
        
        majority_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else None
        
        segment_speakers = []
        for seg in segments:
            speaker_times = {}
            for w in word_speakers:
                if w["speaker"] is not None and w["end"] > seg["start"] and w["start"] < seg["end"]:
                    overlap = min(seg["end"], w["end"]) - max(seg["start"], w["start"])
                    if overlap > 0:
                        speaker_times[w["speaker"]] = speaker_times.get(w["speaker"], 0) + overlap
            dominant_speaker = max(speaker_times, key=speaker_times.get) if speaker_times else None
            segment_speakers.append(dominant_speaker)
        
        t3 = time.time()
        prompt = """
            You are an expert transcript editor. Refine each segment individually with these strict rules:
            1.  Preserve the exact number of segments. Do not merge, split, or delete any.
            2.  Only remove filler words like "um", "uh", "like", "you know". Do not rephrase.
            3.  Correct obvious spelling or grammar mistakes.
            4.  If a segment is just a filler word (e.g., "Okay."), keep it.
            5.  Output exactly one line per input segment, starting with the original timestamp, a space, then the refined text.
            Example Input: 02:45 I, uh, think we should go.
            Example Output: 02:45 I think we should go.
            ---
            Now refine the following segments:
            """
        
        rows = [["Start", "End", "New Start", "New End", "Original", "Refined", "Pause at end(sec)", "Audio Length", "Video Length", "Flag", "Clone Voice"]]
        prompt_text_to_refine = ""
        current_t = 0
        for i, seg in enumerate(segments):
            if seg["start"] - current_t > 0.2:
                rows.append([current_t, seg['start'], current_t, seg['start'], "", "", 0, seg['start'] - current_t, seg['start'] - current_t, "", "yes"])
            clone_voice = "yes" if i >= len(segment_speakers) or segment_speakers[i] is None or segment_speakers[i] == majority_speaker else "no"
            rows.append([seg['start'], seg['end'], "", "", seg['text'].strip(), "", 0, "", seg['end'] - seg['start'], "", clone_voice])
            prompt_text_to_refine += f"{seg['start']} {seg['text'].strip()}\n"
            current_t = seg['end']
        
        refined_text = call_gemini(prompt + prompt_text_to_refine)
        refined_lines = refined_text.splitlines()
        
        refined_map = {}
        for line in refined_lines:
            match = re.match(r"(\d+\.?\d*)\s*(.*)", line)
            if match:
                refined_map[float(match.group(1))] = match.group(2)
        
        for row in rows[1:]: # Skip header
             if isinstance(row[0], (int, float)) and row[0] in refined_map:
                row[5] = refined_map[row[0]]
        log_performance("Gemini Refinement", t3)
        
        t4 = time.time()
        sheet = sheets_service.spreadsheets().create(body={'properties': {'title': filename}}, fields='spreadsheetId').execute()
        sheet_id = sheet['spreadsheetId']
        drive_service.permissions().create(fileId=sheet_id, body={'type': 'anyone', 'role': 'writer'}).execute()
        sheets_service.spreadsheets().values().update(spreadsheetId=sheet_id, range='A1', valueInputOption='RAW', body={'values': rows}).execute()
        sheets_service.spreadsheets().values().update(spreadsheetId=sheet_id, range='M1', valueInputOption='RAW', body={'values': [[filename]]}).execute()
        log_performance("Google Sheet Creation", t4)

        t5 = time.time()
        upload_file(local_video_path, f"Original_videos/{filename}")
        log_performance("Upload Video to S3", t5)

        log_performance("Total /process-video", t_start)
        return JSONResponse({"spreadsheetId": sheet_id, "SpreadsheetUrl": f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit"})

    except Exception as e:
        logger.exception("Error in /process-video endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        # Cleanup *select* temporary files, preserving the cache
        try:
            # We want to delete the large, single files
            files_to_delete = [
                local_video_path,   # The downloaded original video
                final_video_path,   # The generated final video
                final_audio_path,   # The stitched final audio
                ass_path            # The generated caption file
            ]
            
            for f in files_to_delete:
                if f and os.path.exists(f):
                    os.remove(f)
                    logger.info(f"Cleaned up temp file: {f}")
            
            # We INTENTIONALLY DO NOT delete:
            # - local_dir (the main folder)
            # - state_csv_path (the cache state)
            # - cloned_audio_dir (the cached .wav files)

        except Exception as e:
            logger.error(f"Error during selective cleanup: {e}")
        


# =========================================================================================================
#                    ENDPOINT 2: /refresh-voiceover <--- MAJOR CHANGES HERE
# =========================================================================================================
# @app.post("/refresh-voiceover")
# async def refresh_voiceover(sheetId: str): # <--- API Contract: Uses query parameter
#     t_start = time.time()
#     local_dir = None
    
#     # --- CORRECTED: Initialize paths to None for robust cleanup ---
#     local_video_path = None
#     final_video_path = None
#     final_audio_path = None
#     ass_path = None
#     state_csv_path = None # Also initialize this for safety

#     try:
#         t0 = time.time()
#         # --- Setup: Get filename, create directories ---
#         try:
#             filename_resp = sheets_service.spreadsheets().values().get(spreadsheetId=sheetId, range='M1').execute()
#             filename = filename_resp['values'][0][0]
#         except (HttpError, KeyError, IndexError):
#             raise HTTPException(status_code=400, detail="Could not retrieve filename from sheet cell M1.")

#         uid = filename.split('_')[-1].split('.')[0]
#         local_dir = f"./Data/tmp/{uid}"
#         os.makedirs(local_dir, exist_ok=True)
        
#         # --- Assign paths ---
#         local_video_path = os.path.join(local_dir, filename)
#         cloned_audio_dir = os.path.join(local_dir, "cloned")
#         os.makedirs(cloned_audio_dir, exist_ok=True)
#         ass_path = os.path.join(local_dir, "captions.ass")
#         state_csv_path = os.path.join(local_dir, "state.csv") # <-- Assign state path
            
#         download_file(f"Original_videos/{filename}", local_video_path)
#         log_performance("Setup & Download", t0)
        
#         # +++ THIS BLOCK WAS MISSING +++
#         # --- Reference Audio (Use a short clip!) ---
#         # ⚠️ IMPORTANT: Update this path to your short reference audio clip
#         base_dir = os.path.abspath(os.path.dirname(__file__))
#         ref_audio_path = os.path.join(base_dir, "reference_audio.wav") # Assumes Trump-1.wav is in the same folder
        
#         if not os.path.exists(ref_audio_path):
#             logger.error(f"FATAL: Reference audio not found at {ref_audio_path}")
#             raise HTTPException(status_code=500, detail="Reference audio file is missing from the server.")
#         # +++ END OF MISSING BLOCK +++

#         # +++ START CACHING LOGIC +++
#         prev_state = {}
#         if os.path.exists(state_csv_path):
#             logger.info(f"Loading state from {state_csv_path}")
#             try:
#                 df = pandas.read_csv(state_csv_path)
#                 for _, row in df.iterrows():
#                     idx = int(row['Index'])
#                     prev_state[idx] = {
#                         'refined': str(row['Refined']),
#                         'pause': str(row['Pause']),
#                         'CloneVoice': str(row.get('CloneVoice', 'yes'))  
#                     }
#             except Exception as e:
#                 logger.error(f"Error loading state CSV: {e}")
#         # +++ END CACHING LOGIC +++

#         t1 = time.time()
#         rows = sheets_service.spreadsheets().values().get(spreadsheetId=sheetId, range='A2:L').execute().get('values', [])
        
#         # --- RunPod TTS Generation (Concurrent with Caching) ---
#         if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_URL:
#              raise HTTPException(status_code=500, detail="RunPod API Key or Endpoint URL not configured.")
             
#         tasks = []
#         semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNPOD_REQUESTS)
        
#         async with aiohttp.ClientSession() as session:
#             for idx, row in enumerate(rows):
#                 while len(row) <= 10: row.append("") 
#                 refined = row[5]
#                 pause = row[6] if row[6] else '0'
#                 clonevoice = row[10] if row[10] else 'yes'
                
#                 # --- Path to the segment's audio file ---
#                 output_audio_path = os.path.join(cloned_audio_dir, f"seg_{idx}.wav")

#                 unchanged = False
#                 if idx in prev_state:
#                     prev = prev_state[idx]
#                     unchanged = (str(refined) == str(prev['refined']) and 
#                                  str(pause) == str(prev['pause']) and 
#                                  str(clonevoice) == str(prev.get('CloneVoice', 'yes')))

#                 if unchanged:
#                     logger.info(f"[Cache] Segment {idx} is UNCHANGED. Skipping.")
#                     continue # <-- Skip to the next loop iteration

#                 # --- If we are here, the segment has CHANGED ---
#                 logger.info(f"[Cache] Segment {idx} is NEW or CHANGED.")

#                 # +++ THE FIX +++
#                 # Delete the old, stale audio file, no matter what.
#                 # This forces the code to re-generate or re-extract it.
#                 if os.path.exists(output_audio_path):
#                     logger.info(f"Deleting stale audio file: {output_audio_path}")
#                     os.remove(output_audio_path)
#                 # +++ END OF FIX +++

#                 if refined and clonevoice.lower() == "yes":
#                     logger.info(f"[RunPod] Generating new audio for segment {idx}.")
#                     task = asyncio.create_task(
#                         call_runpod_tts(session, semaphore, text=refined, ref_audio_path=ref_audio_path, output_path=output_audio_path, segment_index=idx)
#                     )
#                     tasks.append(task)
                

#             logger.info(f"Starting {len(tasks)} RunPod TTS generations...")
#             results = await asyncio.gather(*tasks)
#             successful_tasks = sum(1 for res in results if res is True)
#             failed_tasks = len(tasks) - successful_tasks
#             logger.info(f"RunPod TTS generation complete. Success: {successful_tasks}, Failed: {failed_tasks}")
#             if failed_tasks > 0:
#                  logger.error(f"{failed_tasks} TTS segments failed to generate via RunPod.")
        
#         log_performance("RunPod TTS Generation (Concurrent)", t1)
        
#         # --- Timestamp calculation and segment processing ---
#         t2 = time.time()
#         segments = []
#         current_new_start = 0.0
#         for idx, row in enumerate(rows):
#             try:
#                 while len(row) <= 10: row.append("") 
#                 start = float(row[0]) if row[0] else 0.0
#                 end = float(row[1]) if row[1] else start
#                 refined = row[5]
#                 pause = float(row[6]) if row[6] else 0.0
#                 clone_voice = row[10]
#                 video_len = max(0, end - start)
#                 audio_path = None
#                 factor = 1.0
#                 audio_len_sec = video_len

#                 if clone_voice.lower() == "no":
#                     audio_path = os.path.join(cloned_audio_dir, f"seg_{idx}.wav")
#                     if not os.path.exists(audio_path) and video_len > 0:
#                          run_ffmpeg(["ffmpeg", "-y", "-ss", str(start), "-i", local_video_path, "-t", str(video_len), "-q:a", "0", "-map", "a", audio_path], f"Audio Extraction for seg {idx}")
#                     elif not os.path.exists(audio_path) and video_len <= 0:
#                          run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.01", "-acodec", "pcm_s16le", audio_path], f"Silent Audio for seg {idx}")
                         
#                 elif refined:
#                     audio_path = os.path.join(cloned_audio_dir, f"seg_{idx}.wav")
#                     if os.path.exists(audio_path):
#                         audio_len_sec = get_media_duration(audio_path) + pause
#                         if audio_len_sec > 0 and video_len > 0:
#                             factor = min(video_len / audio_len_sec, 2.0)
#                         elif video_len <= 0:
#                              factor = float('inf')
#                              audio_len_sec = 0
#                     else:
#                         logger.warning(f"Cloned audio for seg {idx} not found. Using original timing.")
#                         audio_len_sec = video_len

#                 segment_duration = (video_len / factor) if factor > 0 and factor != float('inf') else 0.0
#                 new_end = current_new_start + segment_duration
#                 row[2], row[3] = f"{current_new_start:.3f}", f"{new_end:.3f}"
#                 current_new_start = new_end
#                 row[7] = f"{audio_len_sec:.3f}"
#                 row[8] = f"{video_len:.3f}"
#                 segments.append({"start": start, "end": end, "factor": factor, "audio_path": audio_path, "path": local_video_path})
#             except (ValueError, IndexError, TypeError) as e:
#                 logger.warning(f"Skipping malformed row {idx+2} during segment processing: {row} | Error: {e}")

#         log_performance("Timestamp Calculation", t2)
        
#         # --- Save current state to CSV ---
#         state_data = []
#         for idx, row in enumerate(rows):
#             state_data.append({
#                 "Index": idx,
#                 "Refined": row[5] if len(row) > 5 else "",
#                 "Pause": row[6] if len(row) > 6 else "0",
#                 "CloneVoice": row[10] if len(row) > 10 else "yes"  
#             })
#         state_df = pandas.DataFrame(state_data)
#         logger.info(f"Saving new state to {state_csv_path}")
#         state_df.to_csv(state_csv_path, index=False)
        
#         # --- Final video/audio processing ---
#         t3 = time.time()
#         final_video_path = os.path.join(local_dir, f"final_{filename}") # <--- Assignment
#         processed_path, final_audio_path = process_segments_with_ffmpeg(segments, local_video_path, final_video_path, ass_path, local_dir) # <--- Assignment
#         log_performance("Final Video/Audio Processing", t3)

#         # --- Update Google Sheet ---
#         t4 = time.time()
#         rows_str = [[str(cell) for cell in r] for r in rows]
#         sheets_service.spreadsheets().values().update(spreadsheetId=sheetId, range='A2', valueInputOption='RAW', body={'values': rows_str}).execute()
#         log_performance("Google Sheet Update", t4)

#         # --- Upload final files ---
        
      
#         # # --- Save Final Video Locally ---
#         # t6 = time.time()
#         # local_save_dir = "./final_local_videos" 
#         # os.makedirs(local_save_dir, exist_ok=True)
#         # local_final_path = os.path.join(local_save_dir, f"LOCAL_{filename}")
#         # try:
#         #     shutil.copy2(processed_path, local_final_path)
#         #     logger.info(f"Successfully saved final video locally to: {local_final_path}")
#         # except Exception as e_copy:
#         #     logger.error(f"Failed to save final video locally: {e_copy}")
#         # log_performance("Save Final Video Locally", t6)
        
#         t5 = time.time()
        
#         # ✅ FIX: Capture the upload result to get the URL
#         final_video_upload_result = upload_file(processed_path, f"Final_videos/{filename}")
#         final_audio_upload_result = upload_file(final_audio_path, f"Final_audio/{uid}.wav")
        
#         # ✅ FIX: Extract the URL from the upload result
#         final_s3_url = final_video_upload_result["url"]
        
#         logger.info(f"Final video uploaded to: {final_s3_url}")
#         log_performance("Upload Final Assets to S3", t5)
        
#         log_performance("Total /refresh-voiceover", t_start)
        
#         # ✅ FIX: Return the URL that frontend expects
#         return JSONResponse({
#             "message": "Refresh completed successfully", 
#             "processed_video": filename,
#             "Final_s3_url": final_s3_url  # ← Frontend needs this!
#         })


#     except Exception as e:
#         logger.exception("Error in /refresh-voiceover endpoint")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
#     finally:
#         # --- CORRECTED, ROBUST selective cleanup ---
#         logger.info("Starting selective cleanup...")
#         try:
#             files_to_delete = [
#                 local_video_path,   # The downloaded original video
#                 final_video_path,   # The generated final video
#                 final_audio_path,   # The stitched final audio
#                 ass_path            # The generated caption file
#             ]
            
#             for f_path in files_to_delete:
#                 # Check if variable was assigned AND file exists
#                 if f_path and os.path.exists(f_path):
#                     os.remove(f_path)
#                     logger.info(f"Cleaned up temp file: {f_path}")
            
#             # We INTENTIONALLY DO NOT delete:
#             # - local_dir (the main folder)
#             # - state_csv_path (the cache state)
#             # - cloned_audio_dir (the cached .wav files)

#         except Exception as e:
#             logger.error(f"Error during selective cleanup: {e}")

@app.post("/refresh-voiceover")
async def refresh_voiceover(sheetId: str): 
    t_start = time.time()
    local_dir = None
    
    # --- CORRECTED: Initialize paths to None for robust cleanup ---
    local_video_path = None
    final_video_path = None
    final_audio_path = None
    ass_path = None
    state_csv_path = None 

    try:
        t0 = time.time()
        # --- Setup: Get filename, create directories ---
        try:
            filename_resp = sheets_service.spreadsheets().values().get(spreadsheetId=sheetId, range='M1').execute()
            filename = filename_resp['values'][0][0]
        except (HttpError, KeyError, IndexError):
            raise HTTPException(status_code=400, detail="Could not retrieve filename from sheet cell M1.")

        uid = filename.split('_')[-1].split('.')[0]
        local_dir = f"./Data/tmp/{uid}"
        os.makedirs(local_dir, exist_ok=True)
        
        # --- Assign paths ---
        local_video_path = os.path.join(local_dir, filename)
        cloned_audio_dir = os.path.join(local_dir, "cloned")
        os.makedirs(cloned_audio_dir, exist_ok=True)
        ass_path = os.path.join(local_dir, "captions.ass")
        state_csv_path = os.path.join(local_dir, "state.csv") 
            
        download_file(f"Original_videos/{filename}", local_video_path)
        log_performance("Setup & Download", t0)
        
        # --- Reference Audio (Use a short clip!) ---
        base_dir = os.path.abspath(os.path.dirname(__file__))
        ref_audio_path = os.path.join(base_dir, "reference_audio.wav") 
        
        if not os.path.exists(ref_audio_path):
            logger.error(f"FATAL: Reference audio not found at {ref_audio_path}")
            raise HTTPException(status_code=500, detail="Reference audio file is missing from the server.")
        
        # +++ FIX START: Encode Audio ONCE here +++
        ref_audio_b64 = ""
        try:
            with open(ref_audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                ref_audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            logger.info("Reference audio encoded successfully (Once).")
        except Exception as e:
            logger.error(f"Failed to encode reference audio: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to encode reference audio: {e}")
        # +++ FIX END +++

        # +++ START CACHING LOGIC +++
        prev_state = {}
        if os.path.exists(state_csv_path):
            logger.info(f"Loading state from {state_csv_path}")
            try:
                df = pandas.read_csv(state_csv_path)
                for _, row in df.iterrows():
                    idx = int(row['Index'])
                    prev_state[idx] = {
                        'refined': str(row['Refined']),
                        'pause': str(row['Pause']),
                        'CloneVoice': str(row.get('CloneVoice', 'yes'))  
                    }
            except Exception as e:
                logger.error(f"Error loading state CSV: {e}")
        # +++ END CACHING LOGIC +++

        t1 = time.time()
        rows = sheets_service.spreadsheets().values().get(spreadsheetId=sheetId, range='A2:L').execute().get('values', [])
        
        # --- RunPod TTS Generation (Concurrent with Caching) ---
        # if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_URL:
        #      raise HTTPException(status_code=500, detail="RunPod API Key or Endpoint URL not configured.")
             
        # tasks = []
        # semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNPOD_REQUESTS)
        
        # async with aiohttp.ClientSession() as session:
        #     for idx, row in enumerate(rows):
        #         while len(row) <= 10: row.append("") 
        #         refined = row[5]
        #         pause = row[6] if row[6] else '0'
        #         clonevoice = row[10] if row[10] else 'yes'
                
        #         # --- Path to the segment's audio file ---
        #         output_audio_path = os.path.join(cloned_audio_dir, f"seg_{idx}.wav")

        #         unchanged = False
        #         if idx in prev_state:
        #             prev = prev_state[idx]
        #             unchanged = (str(refined) == str(prev['refined']) and 
        #                          str(pause) == str(prev['pause']) and 
        #                          str(clonevoice) == str(prev.get('CloneVoice', 'yes')))

        #         if unchanged:
        #             logger.info(f"[Cache] Segment {idx} is UNCHANGED. Skipping.")
        #             continue 

        #         # --- If we are here, the segment has CHANGED ---
        #         logger.info(f"[Cache] Segment {idx} is NEW or CHANGED.")

        #         if os.path.exists(output_audio_path):
        #             logger.info(f"Deleting stale audio file: {output_audio_path}")
        #             os.remove(output_audio_path)

        #         if refined and clonevoice.lower() == "yes":
        #             logger.info(f"[RunPod] Generating new audio for segment {idx}.")
        #             # +++ FIX: Pass the pre-encoded 'ref_audio_b64' string here +++
        #             task = asyncio.create_task(
        #                 call_runpod_tts(session, semaphore, text=refined, ref_audio_b64=ref_audio_b64, output_path=output_audio_path, segment_index=idx)
        #             )
        #             tasks.append(task)
        # --- RunPod TTS Generation (Batched & Concurrent) ---
        if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_URL:
             raise HTTPException(status_code=500, detail="RunPod API Key or Endpoint URL not configured.")
             
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNPOD_REQUESTS)
        
        async with aiohttp.ClientSession() as session:
            pending_tasks = []
            
            # 1. PREPARE TASKS (Do not run them yet)
            for idx, row in enumerate(rows):
                while len(row) <= 10: row.append("") 
                refined = row[5]
                pause = row[6] if row[6] else '0'
                clonevoice = row[10] if row[10] else 'yes'
                
                output_audio_path = os.path.join(cloned_audio_dir, f"seg_{idx}.wav")

                # Check Cache
                unchanged = False
                if idx in prev_state:
                    prev = prev_state[idx]
                    unchanged = (str(refined) == str(prev['refined']) and 
                                 str(pause) == str(prev['pause']) and 
                                 str(clonevoice) == str(prev.get('CloneVoice', 'yes')))

                if unchanged:
                    logger.info(f"[Cache] Segment {idx} is UNCHANGED. Skipping.")
                    continue 

                logger.info(f"[Cache] Segment {idx} is NEW or CHANGED.")

                if os.path.exists(output_audio_path):
                    os.remove(output_audio_path)

                if refined and clonevoice.lower() == "yes":
                    # Create the coroutine object, but do not await it yet
                    task = call_runpod_tts(session, semaphore, text=refined, ref_audio_b64=ref_audio_b64, output_path=output_audio_path, segment_index=idx)
                    pending_tasks.append(task)
            
            # 2. EXECUTE IN BATCHES (Prevents Memory Spikes)
            BATCH_SIZE = 5
            total_tasks = len(pending_tasks)
            logger.info(f"Queued {total_tasks} tasks. Processing in batches of {BATCH_SIZE}...")

            for i in range(0, total_tasks, BATCH_SIZE):
                batch = pending_tasks[i : i + BATCH_SIZE]
                logger.info(f"--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch)} tasks) ---")
                
                # Run this batch and wait for it to finish
                results = await asyncio.gather(*batch)
                
                # Count successes in this batch
                success_count = sum(1 for res in results if res is True)
                logger.info(f"Batch {i//BATCH_SIZE + 1} complete. Success: {success_count}/{len(batch)}")
                
                # CRITICAL: Force memory cleanup after every batch
                gc.collect()
                

            logger.info(f"Starting {len(tasks)} RunPod TTS generations...")
            results = await asyncio.gather(*tasks)
            successful_tasks = sum(1 for res in results if res is True)
            failed_tasks = len(tasks) - successful_tasks
            logger.info(f"RunPod TTS generation complete. Success: {successful_tasks}, Failed: {failed_tasks}")
            if failed_tasks > 0:
                 logger.error(f"{failed_tasks} TTS segments failed to generate via RunPod.")
        
        log_performance("RunPod TTS Generation (Concurrent)", t1)
        
        # --- Timestamp calculation and segment processing ---
        t2 = time.time()
        segments = []
        current_new_start = 0.0
        for idx, row in enumerate(rows):
            try:
                while len(row) <= 10: row.append("") 
                start = float(row[0]) if row[0] else 0.0
                end = float(row[1]) if row[1] else start
                refined = row[5]
                pause = float(row[6]) if row[6] else 0.0
                clone_voice = row[10]
                video_len = max(0, end - start)
                audio_path = None
                factor = 1.0
                audio_len_sec = video_len

                if clone_voice.lower() == "no":
                    audio_path = os.path.join(cloned_audio_dir, f"seg_{idx}.wav")
                    if not os.path.exists(audio_path) and video_len > 0:
                         run_ffmpeg(["ffmpeg", "-y", "-ss", str(start), "-i", local_video_path, "-t", str(video_len), "-q:a", "0", "-map", "a", audio_path], f"Audio Extraction for seg {idx}")
                    elif not os.path.exists(audio_path) and video_len <= 0:
                         run_ffmpeg(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.01", "-acodec", "pcm_s16le", audio_path], f"Silent Audio for seg {idx}")
                         
                elif refined:
                    audio_path = os.path.join(cloned_audio_dir, f"seg_{idx}.wav")
                    if os.path.exists(audio_path):
                        audio_len_sec = get_media_duration(audio_path) + pause
                        if audio_len_sec > 0 and video_len > 0:
                            factor = min(video_len / audio_len_sec, 2.0)
                        elif video_len <= 0:
                             factor = float('inf')
                             audio_len_sec = 0
                    else:
                        logger.warning(f"Cloned audio for seg {idx} not found. Using original timing.")
                        audio_len_sec = video_len

                segment_duration = (video_len / factor) if factor > 0 and factor != float('inf') else 0.0
                new_end = current_new_start + segment_duration
                row[2], row[3] = f"{current_new_start:.3f}", f"{new_end:.3f}"
                current_new_start = new_end
                row[7] = f"{audio_len_sec:.3f}"
                row[8] = f"{video_len:.3f}"
                segments.append({"start": start, "end": end, "factor": factor, "audio_path": audio_path, "path": local_video_path})
            except (ValueError, IndexError, TypeError) as e:
                logger.warning(f"Skipping malformed row {idx+2} during segment processing: {row} | Error: {e}")

        log_performance("Timestamp Calculation", t2)
        
        # --- Save current state to CSV ---
        state_data = []
        for idx, row in enumerate(rows):
            state_data.append({
                "Index": idx,
                "Refined": row[5] if len(row) > 5 else "",
                "Pause": row[6] if len(row) > 6 else "0",
                "CloneVoice": row[10] if len(row) > 10 else "yes"  
            })
        state_df = pandas.DataFrame(state_data)
        logger.info(f"Saving new state to {state_csv_path}")
        state_df.to_csv(state_csv_path, index=False)
        
        # --- Final video/audio processing ---
        t3 = time.time()
        final_video_path = os.path.join(local_dir, f"final_{filename}") 
        processed_path, final_audio_path = process_segments_with_ffmpeg(segments, local_video_path, final_video_path, ass_path, local_dir) 
        log_performance("Final Video/Audio Processing", t3)

        # --- Update Google Sheet ---
        t4 = time.time()
        rows_str = [[str(cell) for cell in r] for r in rows]
        sheets_service.spreadsheets().values().update(spreadsheetId=sheetId, range='A2', valueInputOption='RAW', body={'values': rows_str}).execute()
        log_performance("Google Sheet Update", t4)

        # --- Upload final files ---
        t# --- Upload final files ---
        t5 = time.time()
        final_video_key = f"Final_videos/{filename}"
        upload_file(processed_path, final_video_key)
        upload_file(final_audio_path, f"Final_audio/{uid}.wav")
        log_performance("Upload Final Assets to S3", t5)
        
        # --- GENERATE PRESIGNED URL FOR FRONTEND ---
        try:
            final_s3_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': final_video_key},
                ExpiresIn=3600 # Valid for 1 hour
            )
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            final_s3_url = ""
            
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ FINAL S3 URL SENT TO FRONTEND:")
        logger.info(f"{final_s3_url}")
        logger.info(f"{'='*60}\n")


        log_performance("Total /refresh-voiceover", t_start)
        
        # ✅ FIX: Return the URL that frontend expects
        return JSONResponse({
            "message": "Refresh completed successfully", 
            "processed_video": filename,
            "Final_s3_url": final_s3_url  # <--- Included now!
        })

    except Exception as e:
        logger.exception("Error in /refresh-voiceover endpoint")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # --- Clean up ---
        logger.info("Starting selective cleanup...")
        try:
            files_to_delete = [
                local_video_path,   
                final_video_path,   
                final_audio_path,   
                ass_path            
            ]
            
            for f_path in files_to_delete:
                if f_path and os.path.exists(f_path):
                    os.remove(f_path)
                    logger.info(f"Cleaned up temp file: {f_path}")

        except Exception as e:
            logger.error(f"Error during selective cleanup: {e}")






# =========================================================================================================
#                    ENDPOINT 3: /create-avatar-video (HEYGEN)
# =========================================================================================================
# (Remains the same - no changes needed)
async def generate_heygen_video(audio_s3_key: str, avatar_id: str):
    t_start = time.time()
    HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
    if not HEYGEN_API_KEY:
        raise HTTPException(status_code=500, detail="HEYGEN_API_KEY is not set.")

    headers = {"X-Api-Key": HEYGEN_API_KEY, "Content-Type": "application/json"}
    
    # Create a pre-signed URL for Heygen to access the audio from S3
    audio_url = s3.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': audio_s3_key}, ExpiresIn=3600)
    log_performance("Heygen - Generate Presigned URL", t_start)

    t1 = time.time()
    generate_payload = {
        "video_inputs": [{
            "character": {"type": "avatar", "avatar_id": avatar_id, "avatar_style": "normal"},
            "voice": {"type": "audio", "audio_url": audio_url},
            "background": {"type": "color", "value": "#00FF00"} # Green screen
        }],
        "dimension": {"width": 960 , "height": 540}, # Specify dimensions if needed
        "test": True # Use test mode if needed
    }
    
    generate_response = requests.post("https://api.heygen.com/v2/video/generate", headers=headers, json=generate_payload)
    if generate_response.status_code != 200:
        raise HTTPException(status_code=generate_response.status_code, detail=f"Heygen video generation failed to start: {generate_response.text}")
    video_id = generate_response.json()["data"]["video_id"]
    log_performance("Heygen - Start Video Generation", t1)

    t2 = time.time()
    status_url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    video_download_url = None
    # Poll for status (consider increasing timeout or retries)
    for _ in range(60): # Poll for up to 15 minutes (60 * 15 seconds)
        await asyncio.sleep(15) 
        try:
             status_response = requests.get(status_url, headers={"X-Api-Key": HEYGEN_API_KEY}, timeout=30)
             status_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
             status_data = status_response.json().get("data", {})
             current_status = status_data.get('status')
             logger.info(f"Heygen video {video_id} status: {current_status}")
             if current_status in ["succeeded", "completed"]:
                 video_download_url = status_data.get("video_url")
                 if video_download_url:
                     break
                 else:
                     logger.warning(f"Heygen status succeeded but video_url is missing: {status_data}")
                     # Optionally retry or handle this case
             elif current_status == "failed":
                 raise HTTPException(status_code=500, detail=f"Heygen video failed: {status_data.get('error')}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error polling Heygen status: {e}. Retrying...")
        except Exception as e: # Catch other potential errors
             logger.error(f"Unexpected error during Heygen status polling: {e}")
             # Decide if this should be fatal or retried
    
    if not video_download_url:
        raise HTTPException(status_code=500, detail="Heygen video generation timed out or failed to get URL.")
    log_performance("Heygen - Poll for Status", t2)
    return video_download_url

@app.post("/create-avatar-video")
async def create_avatar_video(sheetId: str): # <--- API Contract: NOW ONLY NEEDS sheetId
    t_start = time.time()

    # +++ ADD THIS BLOCK +++
    # Get the filename from the sheet, just like /refresh-voiceover does
    try:
        filename_resp = sheets_service.spreadsheets().values().get(spreadsheetId=sheetId, range='M1').execute()
        filename = filename_resp['values'][0][0]
    except (HttpError, KeyError, IndexError):
        raise HTTPException(status_code=400, detail="Could not retrieve filename from sheet cell M1.")
    # +++ END OF BLOCK +++

    uid = filename.split('_')[-1].split('.')[0]
    local_dir = f"./Data/tmp/{uid}_avatar"
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        t0 = time.time()
        main_video_path = os.path.join(local_dir, filename)
        download_file(f"Final_videos/{filename}", main_video_path)
        log_performance("Avatar - Download Assets", t0)

        t1 = time.time()
        final_audio_s3_key = f"Final_audio/{uid}.wav"
        AVATAR_ID = os.getenv("HEYGEN_AVATAR_ID", "Default_Avatar_ID_If_Not_Set") # Get from env
        avatar_video_url = await generate_heygen_video(final_audio_s3_key, AVATAR_ID)
        
        avatar_video_path = os.path.join(local_dir, "avatar.mp4")
        # Download the generated video
        logger.info(f"Downloading Heygen video from: {avatar_video_url}")
        with requests.get(avatar_video_url, stream=True, timeout=300) as r: # Increased timeout
            r.raise_for_status()
            with open(avatar_video_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                     f.write(chunk)
        log_performance("Avatar - Download Heygen Video", t1)

        t2 = time.time()
        output_path = os.path.join(local_dir, f"avatar_final_{filename}")
        # Make the avatar smaller and place it bottom right
        ffmpeg_filter = (
            "[1:v]scale=-1:240,format=rgba,geq=lum='p(X,Y)':a='if(lte(pow(X-W/2,2)+pow(Y-H/2,2),pow(min(W/2,H/2),2)),255,0)'[avatar_circular];" # Scale to 240px height, make circular
            "[0:v][avatar_circular]overlay=main_w-overlay_w-30:main_h-overlay_h-30" # Place bottom right with 30px margin
        )
        run_ffmpeg(["ffmpeg", "-y", "-i", main_video_path, "-i", avatar_video_path, "-filter_complex", ffmpeg_filter, "-c:a", "copy", output_path], "Avatar Overlay Composition")
        log_performance("Avatar - FFmpeg Composition", t2)
        
        t3 = time.time()
        final_s3_info = upload_file(output_path, f"Avatar_videos/{filename}")
        log_performance("Avatar - Upload Final Video", t3)
        
        log_performance("Total /create-avatar-video", t_start)
        return JSONResponse({"message": "Avatar video created successfully!", "final_video_url": final_s3_info.get("url")})

    except Exception as e:
        logger.exception("Error in /create-avatar-video endpoint")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if local_dir and os.path.exists(local_dir):
            logger.info(f"Cleaning up temporary directory: {local_dir}")
            shutil.rmtree(local_dir)

# Add a root endpoint for basic check
@app.get("/")
def read_root():
    return {"message": "Audio Enhancer API is running"}

# If running directly (e.g., uvicorn index_runpod:app --reload)
# Note: This part is usually not needed if deploying via Docker/RunPod CMD
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
