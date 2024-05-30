import os
import random
import shutil
import cv2
from datetime import datetime
from PIL import Image
import ffmpeg
from pydub import AudioSegment
from transformers import pipeline
from TTS.api import TTS
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def cleanup_folder(folder, exclude=[]):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file not in exclude:
                file_path = os.path.join(folder, file)
                os.remove(file_path)
    else:
        os.makedirs(folder)

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

def generate_descriptions(frames_folder, user_description):
    print("Generating descriptions for frames...")
    descriptions = []
    context = (
        f"{user_description}\n"
        "Describe the key elements in each frame, focusing on the main actions and subjects. "
        "Avoid irrelevant details like the background or clothing unless necessary for context. "
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

def summarize_descriptions(descriptions, user_description=""):
    print("Summarizing descriptions...")
    concatenated_text = user_description + " " + " ".join(descriptions)
    max_length = min(len(concatenated_text) // 2, 150)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a sports commentator prepared to describe legendary boxers in interviews, general demeanor, and matches. Everything your say will be text to speech over a short form video. Only speak as the video narrator"},
            {"role": "user", "content": concatenated_text}
        ],
        model="gpt-4o"
    )
    summary = response.choices[0].message.content
    return summary

def generate_tts_for_summary(summary, tts_output_folder, rap_beats_folder):

    os.makedirs(tts_output_folder, exist_ok=True)
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    tts_output_path = os.path.join(tts_output_folder, "summary_tts.wav")
    try:
        print(f"Generating TTS for summary: {summary}")
        tts.tts_to_file(text=summary, file_path=tts_output_path)
    except RuntimeError as e:
        print(f"Error during TTS generation: {e}")
        return False
   
    beat_files = [file for file in os.listdir(rap_beats_folder) if file.endswith('.wav')]
    if not beat_files:
        print("No beat files found in the directory.")
        return False
    

    beat_path = os.path.join(rap_beats_folder, random.choice(beat_files))
    

    beat_audio = AudioSegment.from_wav(beat_path)
    beat_wav_path = os.path.join(tts_output_folder, "selected_beat.wav")
    beat_audio.export(beat_wav_path, format="wav")

    tts_audio = AudioSegment.from_wav(tts_output_path)
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
    
    return True

def create_final_clip(video_path, tts_output_folder, project_folder):
    print("Creating final clip...")
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)
    
    tts_path = os.path.join(tts_output_folder, "combined_summary.mp3")
    if not os.path.exists(tts_path):
        print(f"TTS output file not found: {tts_path}")
        return
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_name = f"{video_name}_{timestamp}.mp4"
    final_output_path = os.path.join(project_folder, output_video_name)
    final_output_temp_path = os.path.join(project_folder, "final_temp_clip.mp4")

   
    if os.path.exists(final_output_temp_path):
        os.remove(final_output_temp_path)

 
    tts_audio = AudioSegment.from_mp3(tts_path)
    tts_duration = tts_audio.duration_seconds

   
    (
        ffmpeg
        .input(video_path, ss=0, t=tts_duration)
        .filter('crop', 'in_h*9/16', 'in_h')
        .filter('scale', 720, 1280)  # Portrait mode resolution
        .output(final_output_temp_path, vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )
    
    video = ffmpeg.input(final_output_temp_path)
    audio = ffmpeg.input(tts_path)

    (
        ffmpeg
        .output(video, audio, final_output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
        .run()
    )

    print(f'Final clip saved to "{final_output_path}"')


def main(source_folder, old_source_folder, project_folder, user_description=""):
    if not os.path.exists(old_source_folder):
        os.makedirs(old_source_folder)
        
    for video_file in os.listdir(source_folder):
        video_path = os.path.join(source_folder, video_file)
        print(f"Processing video: {video_path}")
        
        frames_folder = 'frames'
        tts_output_folder = 'tts_outputs'
        rap_beats_folder = 'rap_beats'
        
        cleanup_folder(frames_folder)
        cleanup_folder(tts_output_folder)
        extract_frames(video_path, frames_folder)
        
        descriptions = generate_descriptions(frames_folder, user_description)
        summary = summarize_descriptions(descriptions, user_description)
        summary_path = os.path.join(tts_output_folder, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        tts_success = generate_tts_for_summary(summary, tts_output_folder, rap_beats_folder)
        if not tts_success:
            print(f"Skipping video due to TTS error: {video_path}")
            continue
        
        create_final_clip(video_path, tts_output_folder, project_folder)
        shutil.move(video_path, os.path.join(old_source_folder, video_file))

if __name__ == "__main__":
    source_folder = input("Enter the path to the source material folder: ")
    old_source_folder = "old_source_material"
    project_folder = "project_final_clips"
    user_description = input("Enter a brief description of the video content (optional): ")
    main(source_folder, old_source_folder, project_folder, user_description)

