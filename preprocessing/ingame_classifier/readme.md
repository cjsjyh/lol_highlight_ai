## Prerequisites
Installation
```
install -y pytorch torchvision -c pytorch
conda install -c conda-forge matplotlib
pip3 install opencv-python
pip3 install moviepy
```

Create folders for output
```
mkdir inference_result/
mkdir inference_result/temp
mkdir full_video
```

Set path for following lines
```
// ingame_extractor.py (line 5)
MODEL_PATH = "./result_model/final_model.pth"                    // path to VGGNet model for inference
FULL_RAW_PATH = "/home/lol/lol_highlight_ai/preprocessing/downloader/full_raw/" // path to full videos
CLASSIFIER_PATH = "/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/" // path to current dir. (some libraries don't support relative path)

// util.py (line5)
HIGHLIGHT_VIDEO_PATH = '/home/lol/total_highlight_video/' // Specify directory where highlight videos are saved
```

## Execution
```
python3 ingame_extractor.py
```

## Manual Jobs
After running `ingame_extractor.py`, `exceptions.txt` will be made.

Games in this list have different number of games compared to highlight videos
due to highlight videos.

Navigate to `ingame_classifier/inference_result` and manually fix the inference
result by deleting unwanted termination timestamps due to pauses.
