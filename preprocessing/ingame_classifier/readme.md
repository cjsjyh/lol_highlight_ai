## Prerequisites
install -y pytorch torchvision -c pytorch
conda install -c conda-forge matplotlib
pip3 install opencv-python
pip3 install moviepy

mkdir inference_result/
    -> mkdir inference_result/temp

mkdir full_video


## Manual Jobs
After running `ingame_extractor.py`, `exceptions.txt` will be made.

Games in this list have different number of games compared to highlight videos
due to highlight videos.

Navigate to `ingame_classifier/inference_result` and manually fix the inference
result by deleting unwanted termination timestamps due to pauses.
