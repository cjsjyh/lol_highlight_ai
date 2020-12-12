Current Codes Are based on VASNet, which is "github.com/ok1zjf/VASNet"   
How to Run the Whole Process :   

1. Preprocessing. - use preprocessing directory   
(1) download dataset (use preprocessing/downloader)   
(2) Highlight Videos : create total_highlight_video/ collecting all highlight videos you want   
(3) Full Videos : before create total_full_video, you need to separate twitch videos by match. (preprocessing/ingame_classifier)   
(4) Videos -> h5 dataset : run video_to_data.py   
    We used 6 processes. However, sometime processes were killed lack of memory.   
    We recommand to use 3 processes at the same time.   
    (python video_to_data -i 0 -v total_full_video/ -d dataset)   
    after that, merge all separated h5 to one using preprocessing/h5   
(*) Sometimes, new features are needed on merged h5 files. in that case, run feature_merge.py on proprocessing/h5. Otherwise, read wideresnet_to_data.py, effnet_to_data.py, audio_to_data.py as example and follow it.       
(5) Labeling whether a scene is highlight or not is necessary. use preprocessing/labeling   
(6) When labeling is done, output.txt will be generated. Then run target_data_insertion to insert labeling information to h5      

Now, You are ready to run the model!   


2. Training - use model directory   
(1) move your dataset to lol_highlight_ai/model/datasets   
(2) run create_split.py -d datasets/yourdatset.h5 --num-splits 5 --save-dir splits  
(3) change splits.json to [yourdataset]_splits.json  
(4) open config.py  
(5) on self.datasets, add your dataset's path  
(6) on self.splits, add your split's path  
(7) if you added new model, import it on config.py and add it's name on self.model_name  
(8) run training session : python main.py -t -m [model_name] -o [path to save your result]   
(9) run evalution session : python main.py -t -m [model_name] -o [path that u saved your result]   
(10) run precision.py : python precision.py -i [idx_num] -d [dataset]    
(11) create machine highlighted video : python machine_sum.py -i [idx_num] -d [dataset]    

However, you can do (8)-(11) automatically by running test_and_val.sh
(8~11) run ./test_and_val.sh
    "set name of model to run : " (insert the model's name you set on self.mode_name)   
    "give name of data except .h5 : " (if the dataset's name is abc.h5, insert abc)    
    "set name of experiment's log : " (set the record for running)   
        * if you run multiple main.py with same datasets, log may be overwritten   
    "set name of result_file : " (set the result directory)   


