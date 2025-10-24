from argparse import ArgumentParser
from pathlib import Path

from moviepy import VideoFileClip

def main():
    parser = ArgumentParser(description="change video fps")
    parser.add_argument("source", help="folder with all video files")
    parser.add_argument("dest", help="folder with all video files")
    parser.add_argument( "fps", help="new fps")
    args = parser.parse_args()
    # must be consistent with the label studio fps
    source = Path(args.source)
    dest = Path(args.dest)

    for mp4_file in source.glob("*.mp4"):
        video_clip = VideoFileClip(mp4_file)

        video_clip.write_videofile(dest / mp4_file.name, fps = int(args.fps), codec='libx264', audio_codec='aac')
        video_clip.close()

if __name__ == "__main__":
    main()
