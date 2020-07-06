import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="videos.txt", help="Video url list file name")
parser.add_argument('-o', type=str, default="download", help="quality or download") 
parser.add_argument('-q', type=str, default="480p", help="video download qualtiy")

args = parser.parse_args()

f = open(args.f,"r")
while True:
	line = f.readline()
	if not line: break

	video_url = line.strip()
	if args.o == "quality":
		subprocess.call(f"youtube-dl -F {video_url}", shell=True)
	if args.o == "download":
		subprocess.call(f"youtube-dl -f {args.q} {video_url}", shell=True)
	print(line.strip())
f.close()
