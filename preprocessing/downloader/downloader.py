import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="videos.txt", help="Video url list file name")
parser.add_argument('-o', type=str, default="download", help="quality or download") 
parser.add_argument('-q', type=str, default="480p", help="video download qualtiy")

args = parser.parse_args()

f = open(args.f,"r")
while True:
    title = f.readline()
    timestamp = f.readline()
    url = f.readline()
    if not title: break

    title_split = title.split()
    filename = ""
    for index, word in enumerate(title_split):
        if word == "vs" or word == "vs." or word == ".vs":
            team1 = title_split[index-1]
            team2 = title_split[index+1]
            filename += f'_{team1}_{team2}'
    if filename != "":
        timestamp_split = timestamp.split()
        datestring = timestamp_split[1].replace('-','')
        datestring = "".join(datestring)
        filename = datestring + filename
        print(filename)
    else:
        print("Unhandled title: " + title)
        break

    video_url = url.strip()
    if args.o == "quality":
        subprocess.call(f"youtube-dl -F {video_url}", shell=True)
    if args.o == "download":
        subprocess.call(f'youtube-dl -f {args.q} -o "full_raw/{filename}.mp4" {video_url}', shell=True)
    print(title)
f.close()

