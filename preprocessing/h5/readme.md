h5 dataset is made by video_to_data.py files
by using video_to_data, a,b,c,d,e,f .h5 files are created.
a,b,c,d,e,f are separated files since using multiple process is much faster.
merge_h5.py and feature_merge_h5.py are used when you need to merge h5 files or
insert new features.
These are really short code so you probably understand directly how to use them.

in short.
1. run video_to_data.py (ex. video_to_data -i 0 -d total_full_video/ -d h5/a.h5)
2. after a,b,c,d,e,f are created, run merge_h5.py
3. if you want to copy features from another h5, use feature_merge_h5.py

