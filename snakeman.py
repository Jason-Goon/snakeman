import os
import random
import yt_dlp as youtube_dl
import cv2
from PIL import Image
import ffmpeg
from pydub import AudioSegment
from transformers import pipeline
from TTS.api import TTS
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the image-to-text model
image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

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
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict.get('title', 'Boxing Match')

# Progress hook for download feedback
def download_progress_hook(d):
    if d['status'] == 'downloading':
        print(f"Downloading: {d['_percent_str']} of {d['_total_bytes_str']} at {d['_speed_str']}")
    elif d['status'] == 'finished':
        print(f"Download completed, converting {d['filename']}")

# Function to extract frames from video for analysis
def extract_frames(video_path, output_folder, interval=20):
    print("Extracting frames...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(fps * interval)
    
    success, image = cap.read()
    while success:
        if frame_count % interval_frames == 0:
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
        "This is a boxing match featuring legendary boxers. "
        "Describe the movements and actions of the boxers, focusing on their punches, defenses, and strategies. "
        "Emphasize their speed, skill, and charisma. "
        "Avoid irrelevant details like the background or clothing. "
        "Describe each frame as part of a continuous sequence of events."
    )

    for frame in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame)
        image = Image.open(frame_path)
        prompt = f"{context[:500]}\nFrame description:"  # Limit context to 500 characters
        description = image_to_text_model(image, max_new_tokens=50)[0]['generated_text']
        # Filter out irrelevant descriptions
        if not any(irrelevant in description for irrelevant in ["background", "flower", "standing next to", "lion", "wrestling", "flag", "wrestling ring", "white shirt"]):
            descriptions.append(description)
        context += f" {description}"

    return descriptions

# Function to summarize descriptions with GPT-4o
def summarize_descriptions(descriptions, user_description=""):
    print("Summarizing descriptions...")
    concatenated_text = user_description + " " + " ".join(descriptions)
    max_length = min(len(concatenated_text) // 2, 150)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a sports commentator."},
            {"role": "user", "content": concatenated_text}
        ],
        model="gpt-4o"
    )
    summary = response.choices[0].message.content
    return summary

# Function to generate TTS for summary and combine with beats
def generate_tts_for_summary(summary, tts_output_folder, rap_beats_folder):
    # Ensure the output folder exists
    os.makedirs(tts_output_folder, exist_ok=True)
    
    # Initialize TTS
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

    # Generate TTS
    tts_output_path = os.path.join(tts_output_folder, "summary_tts.wav")
    tts.tts_to_file(text=summary, file_path=tts_output_path)
    
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
    tts_audio = AudioSegment.from_wav(tts_output_path)
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
    output_video_path = os.path.join(final_output_folder, "final_clip.mp4")
    final_output_temp_path = os.path.join(final_output_folder, "final_temp_clip.mp4")

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
        .output(video, audio, final_output_temp_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
        .run()
    )

    # Rename the temp file to the final file
    os.rename(final_output_temp_path, output_video_path)

    print(f'Final clip saved to "{final_output_folder}"')

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

    # Download video and get title
    video_title = download_video(video_url, video_path)
    if not user_description:
        user_description = video_title

    # Extract frames
    extract_frames(video_path, frames_folder)

    # Process video
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

