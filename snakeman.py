import os
import random
import yt_dlp as youtube_dl
import cv2
from PIL import Image
import ffmpeg
from pydub import AudioSegment
from transformers import pipeline
from gtts import gTTS

# Disable Tokenizers Parallelism Warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Install PyTorch
try:
    import torch
except ImportError:
    os.system('pip install torch torchvision torchaudio')

# Initialize the image-to-text model and the summarization model
image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to clean up folders
def cleanup_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            os.remove(file_path)
    else:
        os.makedirs(folder)

# Function to download YouTube video
def download_video(youtube_url, output_path):
    ydl_opts = {
        'format': 'best[height<=480]',  # Set video quality to 480p or lower
        'outtmpl': output_path,
        'progress_hooks': [download_progress_hook],  # Hook for download progress
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Progress hook for download feedback
def download_progress_hook(d):
    if d['status'] == 'downloading':
        print(f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}")
    elif d['status'] == 'finished':
        print(f"Download completed, converting {d['filename']}")

# Function to extract frames from video for analysis
def extract_frames(video_path, output_folder, interval=300):
    print("Extracting frames...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, image = cap.read()
    while success:
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame{frame_count}.jpg")
            cropped_image = image[:720, :1280]  # Crop to 720p resolution
            cv2.imwrite(frame_filename, cropped_image)
        success, image = cap.read()
        frame_count += 1
    cap.release()
    print(f"Extracted frames to {output_folder}")

# Function to generate descriptions for frames
def generate_descriptions(frames_folder, user_description):
    print("Generating descriptions for frames...")
    descriptions = []
    context = (
        f"{user_description}\n"
        "This is a boxing match. Describe the movements and actions of the boxers in a detailed, dramatic, and dynamic manner. "
        "Remember to keep track of the actions and avoid repetition. "
        "Consider the context and describe each frame as part of a continuous sequence of events."
    )

    for frame in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame)
        image = Image.open(frame_path)
        prompt = f"{context[:500]}\nFrame description:"  # Limit context to 500 characters
        description = image_to_text_model(image, max_new_tokens=50)[0]['generated_text']
        descriptions.append(description)
        context += f" {description}"

    return descriptions

# Function to summarize descriptions
def summarize_descriptions(descriptions, user_description=""):
    print("Summarizing descriptions...")
    concatenated_text = user_description + " " + " ".join(descriptions)
    # Adjust max_length to be less than or equal to half of the input length to avoid warnings
    max_length = min(len(concatenated_text) // 2, 150)
    summary = summarization_model(concatenated_text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
    return summary

# Function to generate TTS for summary and combine with beats
def generate_tts_for_summary(summary, tts_output_folder, rap_beats_folder):
    # Ensure the output folder exists
    os.makedirs(tts_output_folder, exist_ok=True)
    
    # Generate TTS
    tts_output_path = os.path.join(tts_output_folder, "summary_tts.mp3")
    tts = gTTS(text=summary, lang='en')
    tts.save(tts_output_path)
    
    # Convert TTS output to WAV format
    tts_audio = AudioSegment.from_mp3(tts_output_path)
    tts_wav_path = os.path.join(tts_output_folder, "summary_tts.wav")
    tts_audio.export(tts_wav_path, format="wav")
    
    # List all beats in the folder
    beat_files = [file for file in os.listdir(rap_beats_folder) if file.endswith('.wav')]
    if not beat_files:
        print("No beat files found in the directory.")
        return
    
    # Select a random beat
    beat_path = os.path.join(rap_beats_folder, random.choice(beat_files))
    
    # Convert beat to WAV format
    beat_audio = AudioSegment.from_wav(beat_path)
    beat_wav_path = os.path.join(tts_output_folder, "selected_beat.wav")
    beat_audio.export(beat_wav_path, format="wav")

    # Ensure TTS duration is at least 25 seconds
    tts_duration_ms = tts_audio.duration_seconds * 1000
    if tts_duration_ms < 25 * 1000:
        silence_duration_ms = (25 * 1000) - tts_duration_ms
        silence = AudioSegment.silent(duration=silence_duration_ms)
        tts_audio = tts_audio + silence
        tts_duration_ms = 25 * 1000

    # If more than 60 seconds, trim it to 60 seconds
    if tts_duration_ms > 60 * 1000:
        tts_audio = tts_audio[:60 * 1000]
        tts_duration_ms = 60 * 1000

    beat_audio = beat_audio[:tts_duration_ms]  # Trim beat to match TTS duration
    
    # Combine the TTS and beat audio
    combined = beat_audio.overlay(tts_audio)
    combined_output_path = os.path.join(tts_output_folder, "combined_summary.mp3")
    combined.export(combined_output_path, format="mp3")

# Function to create final clip
def create_final_clip(video_path, tts_output_folder, final_output_folder):
    print("Creating final clip...")
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)
    
    tts_path = os.path.join(tts_output_folder, "combined_summary.mp3")
    output_video_path = os.path.join(final_output_folder, "final_clip_video.mp4")
    final_output_path = os.path.join(final_output_folder, "final_clip.mp4")

    # Get the duration of the TTS audio
    tts_audio = AudioSegment.from_mp3(tts_path)
    tts_duration = tts_audio.duration_seconds

    # Trim the video to the duration of the TTS audio and convert to portrait mode (9:16 aspect ratio)
    (
        ffmpeg
        .input(video_path, ss=0, t=tts_duration)
        .filter('crop', 'in_h*9/16', 'in_h')
        .filter('scale', 720, 1280)  # Portrait mode resolution
        .output(output_video_path, vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )
    
    # Combine the TTS audio with the video
    video = ffmpeg.input(output_video_path)
    audio = ffmpeg.input(tts_path)
    (
        ffmpeg
        .output(video, audio, final_output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
        .run()
    )

    print(f'Final clip saved to "{final_output_path}"')

# Main function
def main(video_url, user_description=""):
    video_path = 'boxing_match.mp4'
    frames_folder = 'frames'
    tts_output_folder = 'tts_outputs'
    final_output_folder = 'final_clips'
    rap_beats_folder = 'rap_beats'

    # Cleanup folders
    cleanup_folder(frames_folder)
    cleanup_folder(tts_output_folder)
    cleanup_folder(final_output_folder)

    # Download video
    download_video(video_url, video_path)

    # Process video
    extract_frames(video_path, frames_folder)
    descriptions = generate_descriptions(frames_folder, user_description)
    summary = summarize_descriptions(descriptions, user_description)
    
    # Save summary to file
    summary_path = os.path.join(tts_output_folder, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    generate_tts_for_summary(summary, tts_output_folder, rap_beats_folder)
    create_final_clip(video_path, tts_output_folder, final_output_folder)

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    user_description = input("Enter a brief description of the video: ")
    main(video_url, user_description)
