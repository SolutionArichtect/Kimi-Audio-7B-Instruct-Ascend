# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import subprocess
import time
from typing import Optional
from cog import BasePredictor, Input, Path, BaseModel
import soundfile as sf

# Define Output class for returning either media or text
class Output(BaseModel):
    media_path: Optional[Path] = None
    json_str: Optional[str] = None

# Setup environment
MODEL_CACHE = "model_cache"
OUTPUT_DIR = "/tmp/output"
SAMPLING_RATE = 24000

# Set cache environment variables
BASE_URL = "https://weights.replicate.delivery/default/kimi-audio-7b-instruct/model_cache/"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

# Try to install and import flash-attn if not already installed
try:
    import flash_attn
    print("flash-attn already installed")
except ImportError:
    print("flash-attn not found, attempting to install...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "flash-attn", "--no-build-isolation"
    ])
    print("flash-attn installed successfully")
   
# Now import KimiAudio after flash-attn handling
from kimia_infer.api.kimia import KimiAudio

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the KimiAudio model into memory"""
        os.makedirs(MODEL_CACHE, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        model_files = [
            "models--THUDM--glm-4-voice-tokenizer.tar",
            "models--moonshotai--Kimi-Audio-7B-Instruct.tar",
            "modules.tar",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Initialize the KimiAudio model
        # NOTE: KimiAudio wrapper handles CUDA placement internally
        self.model = KimiAudio(
            model_path="moonshotai/Kimi-Audio-7B-Instruct",
            load_detokenizer=True,
        )

    def predict(
        self,
        audio: Path = Input(description="Input audio file for processing. Can be used for speech-to-text (ASR) or audio-to-audio generation."),
        prompt: Optional[str] = Input(description="Optional text prompt to guide the model. For ASR, use prompts like 'Please convert this audio to text' or '请将音频内容转换为文字' (Chinese).", default=None),
        output_type: str = Input(description="Type of output to generate: 'audio' for audio only, 'text' for transcription only, or 'both' for both audio and text responses.", choices=["audio", "text", "both"], default="both"),
        return_json: bool = Input(description="Return text results in JSON format instead of text file", default=True),
        # Sampling parameters
        audio_temperature: float = Input(description="Temperature for audio generation. Higher values (0.8-1.0) increase creativity but may reduce coherence.", default=0.8),
        audio_top_k: int = Input(description="Top-k for audio generation. Limits the token selection to the k most likely tokens.", default=10),
        text_temperature: float = Input(description="Temperature for text generation. Lower values (0.0-0.5) increase factual accuracy.", default=0.0),
        text_top_k: int = Input(description="Top-k for text generation. Limits the token selection to the k most likely tokens.", default=5),
        audio_repetition_penalty: float = Input(description="Repetition penalty for audio. Values > 1.0 discourage repetition in audio generation.", default=1.0),
        audio_repetition_window_size: int = Input(description="Window size for audio repetition penalty calculation.", default=64),
        text_repetition_penalty: float = Input(description="Repetition penalty for text. Values > 1.0 discourage repetition in text generation.", default=1.0),
        text_repetition_window_size: int = Input(description="Window size for text repetition penalty calculation.", default=16),
    ) -> Output:
        """Run inference with the KimiAudio model"""
        # Build messages
        messages = []
        if prompt:
            messages.append({"role": "user", "message_type": "text", "content": prompt})
        messages.append({"role": "user", "message_type": "audio", "content": str(audio)})

        # Set sampling parameters
        sampling_params = {
            "audio_temperature": audio_temperature,
            "audio_top_k": audio_top_k,
            "text_temperature": text_temperature,
            "text_top_k": text_top_k,
            "audio_repetition_penalty": audio_repetition_penalty,
            "audio_repetition_window_size": audio_repetition_window_size,
            "text_repetition_penalty": text_repetition_penalty,
            "text_repetition_window_size": text_repetition_window_size,
        }

        # Generate output
        wav, text = self.model.generate(messages, **sampling_params, output_type=output_type)

        # Handle outputs
        audio_path = None
        text_path = None
        
        # Save audio if available
        if output_type in ["audio", "both"] and wav is not None:
            audio_path = os.path.join(OUTPUT_DIR, "output.wav")
            sf.write(audio_path, wav.detach().cpu().view(-1).numpy(), SAMPLING_RATE)
            print(f"Written output to {audio_path}")
        
        # Save text if available
        if output_type in ["text", "both"] and text:
            print(">>> output text: ", text)
            text_path = os.path.join(OUTPUT_DIR, "output.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Written output to {text_path}")
        
        # Create the output object based on what's available and user preferences
        result = Output()
        
        # Handle JSON return option first (simplest case)
        if return_json and text:
            result.json_str = text
            # When returning JSON, also include audio path if available and requested
            if output_type in ["both", "audio"] and audio_path:
                result.media_path = Path(audio_path)
            return result
            
        # Handle media path returns (when not returning JSON)
        # Priority order: audio (for audio or both), text (for text or both), fallback
        if output_type in ["audio", "both"] and audio_path:
            result.media_path = Path(audio_path)
            return result
            
        if output_type in ["text", "both"] and text_path:
            result.media_path = Path(text_path)
            return result
        
        # Fallback case - no valid outputs generated
        if return_json:
            result.json_str = "No output generated"
        else:
            fallback_path = os.path.join(OUTPUT_DIR, "empty_output.txt")
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write("No output generated")
            result.media_path = Path(fallback_path)
            
        return result
