import os
import random
import yt_dlp as youtube_dl
import cv2
from PIL import Image
import ffmpeg
from pydub import AudioSegment
from transformers import pipeline
from gtts import gTTS

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import torch
except ImportError:
    os.system('pip install torch torchvision torchaudio')

image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
summarization_model = pipeline("summarization", model="facebook/bart-large-cnn")

def cleanup_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            os.remove(file_path)
    else:
        os.makedirs(folder)

def download_video(youtube_url, output_path):
    ydl_opts = {
        'format': 'best[height<=480]',
        'outtmpl': output_path,
        'progress_hooks': [download_progress_hook],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict.get('title', 'Boxing Match')

def download_progress_hook(d):
    if d['status'] == 'downloading':
        print(f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}")
    elif d['status'] == 'finished':
        print(f"Download completed, converting {d['filename']}")

def extract_frames(video_path, output_folder, interval=150):
    print("Extracting frames...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, image = cap.read()
    while success:
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame{frame_count}.jpg")
            cropped_image = image[:720, :1280]
            cv2.imwrite(frame_filename, cropped_image)
        success, image = cap.read()
        frame_count += 1
    cap.release()
    print(f"Extracted frames to {output_folder}")

def generate_descriptions(frames_folder, user_description):
    print("Generating descriptions for frames...")
    descriptions = []
    context = (
        f"{user_description}\n"
        "This is a boxing match featuring the legendary Muhammad Ali and his opponent Cleveland Williams. "
        "Describe the movements and actions of the boxers, especially focusing on the skills and prowess of Muhammad Ali. "
        "Avoid mentioning irrelevant details like background or clothing. "
        "Remember to keep track of the actions and avoid repetition. "
        "Consider the context and describe each frame as part of a continuous sequence of events."
    )

    for frame in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame)
        image = Image.open(frame_path)
        prompt = f"{context[:500]}\nFrame description:"
        description = image_to_text_model(image, max_new_tokens=50)[0]['generated_text']
        if not any(irrelevant in description for irrelevant in ["background", "flower", "standing next to", "lion", "wrestling", "flag", "wrestling ring", "white shirt"]):
            descriptions.append(description)
        context += f" {description}"

    return descriptions

def summarize_descriptions(descriptions, user_description=""):
    print("Summarizing descriptions...")
    praise_text = "The legendary Muhammad Ali, known for his unmatched skill and charisma, dominates the ring against Cleveland Williams."
    concatenated_text = praise_text + " " + user_description + " " + " ".join(descriptions)
    max_length = min(len(concatenated_text) // 2, 150)
    summary = summarization_model(concatenated_text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def generate_tts_for_summary(summary, tts_output_folder, rap_beats_folder):
    os.makedirs(tts_output_folder, exist_ok=True)
    
    tts_output_path = os.path.join(tts_output_folder, "summary_tts.mp3")
    tts = gTTS(text=summary, lang='en')
    tts.save(tts_output_path)
    
    tts_audio = AudioSegment.from_mp3(tts_output_path)
    tts_wav_path = os.path.join(tts_output_folder, "summary_tts.wav")
    tts_audio.export(tts_wav_path, format="wav")
    
    beat_files = [file for file in os.listdir(rap_beats_folder) if file.endswith('.wav')]
    if not beat_files:
        print("No beat files found in the directory.")
        return
    
    beat_path = os.path.join(rap_beats_folder, random.choice(beat_files))
    beat_audio = AudioSegment.from_wav(beat_path)
    beat_wav_path = os.path.join(tts_output_folder, "selected_beat.wav")
    beat_audio.export(beat_wav_path, format="wav")

    tts_duration_ms = tts_audio.duration_seconds * 1000
    if tts_duration_ms < 25 * 1000:
        silence_duration_ms = (25 * 1000) - tts_duration_ms
        silence = AudioSegment.silent(duration=silence_duration_ms)
        tts_audio = tts_audio + silence
        tts_duration_ms = 25 * 1000

    if tts_duration_ms > 60 * 1000:
        tts_audio = tts_audio[:60 * 1000]
        tts_duration_ms = 60 * 1000

    beat_audio = beat_audio[:tts_duration_ms]
    
    combined = beat_audio.overlay(tts_audio)
    combined_output_path = os.path.join(tts_output_folder, "combined_summary.mp3")
    combined.export(combined_output_path, format="mp3")

def create_final_clip(video_path, tts_output_folder, final_output_folder):
    print("Creating final clip...")
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)
    
    tts_path = os.path.join(tts_output_folder, "combined_summary.mp3")
    output_video_path = os.path.join(final_output_folder, "final_clip_video.mp4")
    final_output_path = os.path.join(final_output_folder, "final_clip.mp4")

    tts_audio = AudioSegment.from_mp3(tts_path)
    tts_duration = tts_audio.duration_seconds

    (
        ffmpeg
        .input(video_path, ss=0, t=tts_duration)
        .filter('crop', 'in_h*9/16', 'in_h')
        .filter('scale', 720, 1280)
        .output(output_video_path, vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )
    
    video = ffmpeg.input(output_video_path)
    audio = ffmpeg.input(tts_path)
    (
        ffmpeg
        .output(video, audio, final_output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
        .run()
    )

    print(f'Final clip saved to "{final_output_path}"')

def main(video_url, user_description=""):
    video_path = 'boxing_match.mp4'
    frames_folder = 'frames'
    tts_output_folder = 'tts_outputs'
    final_output_folder = 'final_clips'
    rap_beats_folder = 'rap_beats'

    cleanup_folder(frames_folder)
    cleanup_folder(tts_output_folder)
    cleanup_folder(final_output_folder)

    video_title = download_video(video_url, video_path)
    if not user_description:
        user_description = video_title

    extract_frames(video_path, frames_folder)
    descriptions = generate_descriptions(frames_folder, user_description)
    summary = summarize_descriptions(descriptions, user_description)
    
    summary_path = os.path.join(tts_output_folder, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    generate_tts_for_summary(summary, tts_output_folder, rap_beats_folder)
    create_final_clip(video_path, tts_output_folder, final_output_folder)

if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ")
    user_description = input("Enter a brief description of the video: ")
    main(video_url, user_description)
