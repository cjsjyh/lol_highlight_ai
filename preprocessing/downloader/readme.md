## downloader.py
Python script to download twitch / youtube videos from url list file
```
-f :
  input: file name
  description: name of file with video url list
  default: videos.txt
-o : quality / download
  input: quality / download
  description: insert *quality* for video quality output or *download* for downloading videos
  default: download
-q : 
  input: Audio_Only / 160p / 360p / 480p / 720p / 720p60 / 1080p60
  description: specify video quality for download
  default: 480p
```


## prerequisite
ffmpeg
```
// Ubuntu
sudo apt update
sudo apt install ffmpeg

// MacOS
brew update
brew install ffmpeg
```

youtube-dl
```
// Ubuntu
sudo curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl
sudo chmod a+rx /usr/local/bin/youtube-dl

// MacOS
brew install youtube-dl
```

Format of videos.txt
```
SP vs. DRX - kt vs. AF
Published 2020-07-05 @ 07:30:14  Length: 5 h 28 min
https://twitch.tv/videos/12341234
```

## Execution
```
python3 downloader.py
```
`downloader.py` will download games from twitch and save to `/full_raw`



## twitch_list_processor.py
can get a list of LCK videos by tiwtch-dl videos lck.   
after that, make a list from the text to automate download process
