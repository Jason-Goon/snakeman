import os
import time
from googleapiclient.discovery import build
import yt_dlp as youtube_dl
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Set up your API key
API_KEY = 'Replace with your API key'   

# Function to create a source material folder
def create_source_material_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")


def search_youtube_videos(api_key, keyword, max_results=50):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        part='snippet',
        q=keyword,
        maxResults=max_results,
        type='video',
        videoLicense='creativeCommon'
    )
    response = request.execute()
    return [(item['id']['videoId'], item['snippet']['title']) for item in response['items']]


def download_video(video_id, title, output_path):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'best[height<=480]',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'retries': 3,  # Retry up to 3 times on errors
        'noprogress': True,
        'quiet': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading video {title}: {e}")
            return False


def trim_video(input_path, output_path, max_duration=300):
    try:
        ffmpeg_extract_subclip(input_path, 0, max_duration, targetname=output_path)
        print(f"Trimmed video saved as: {output_path}")
        os.remove(input_path)
    except Exception as e:
        print(f"Error trimming video {input_path}: {e}")

def main(api_key, keyword, source_folder, sanity_check_url=None):
    create_source_material_folder(source_folder)

    if sanity_check_url:
        print(f"Performing sanity check with URL: {sanity_check_url}")
        sanity_check_file = os.path.join(source_folder, 'sanity_check.mp4')
        if download_video(sanity_check_url.split('?v=')[-1], 'Sanity Check', sanity_check_file):
            print("Sanity check video downloaded successfully.")
            trim_video(sanity_check_file, sanity_check_file)
        else:
            print("Sanity check video download failed.")

    video_urls = search_youtube_videos(api_key, keyword)

    for video_id, title in video_urls:
        output_file_name = f"{source_folder}/{title.replace(' ', '_')}.mp4"
        if download_video(video_id, title, output_file_name):
            print(f"Successfully downloaded video: {title}")
            trimmed_output_file = output_file_name.replace('.mp4', '_trimmed.mp4')
            trim_video(output_file_name, trimmed_output_file)
        else:
            print(f"Failed to download video: {title}")

if __name__ == "__main__":
    api_key = API_KEY
    keyword = input("Enter the keyword (boxer name) to search for: ")
    source_folder = "source_material"
    sanity_check_url = input("Enter a sanity check URL (leave empty to skip): ").strip() or None
    main(api_key, keyword, source_folder, sanity_check_url)
