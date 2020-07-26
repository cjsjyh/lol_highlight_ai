from os import walk
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

#----------------------
# Common
#----------------------
# Get file names in directory
def get_filenames(path):
  f = []
  for (dirpath, dirnames, filenames) in walk(path):
      f.extend(filenames)
      break
  return f

folder_list = ['2019_wc_highlight_video/',
'2020_promotion_highlight_video/','2020_spring_highlight_video/',
'2020_msc_highlight_video/', '2020_summer_highlight_video/']
def findHighlightVideoPath(video_name, highlight_path):
    for folder_name in folder_list:
        result = glob.glob(f"{highlight_path}{folder_name}{video_name}*")
        if (len(result) != 0):
            return f"{highlight_path}{folder_name}"

#-----------------------
# Get game series
#-----------------------

def get_game_info(highlight_path, full_name):
  full_name_split = full_name.split('.')
  file_ext = full_name_split[1]
  file_name = full_name_split[0]

  if (file_ext != "mp4"):
    return

  file_name_split = file_name.split('_')
  game_date = file_name_split[0]
  # Find last set number for games in 1 video
  game_info = []
  for i in range(1, int(len(file_name_split)/2)+1):
    team1 = file_name_split[i*2 - 1]
    team2 = file_name_split[i*2]
    # Find highlight videos
    filtered_list = glob.glob(f"{highlight_path}{game_date}_{team1}_{team2}*")
    if (len(filtered_list) == 0):
        # Try with reverse order
        filtered_list = glob.glob(f"{highlight_path}{game_date}_{team2}_{team1}*")
        temp = team1
        team1 = team2
        team2 = temp
        if(len(filtered_list) == 0):
            return None
    last_set_num = 0
    for name in filtered_list:
      if (name == full_name):
        break
      else:
        name = name.split('/')[-1]
        name_split = name.split('_')
        set_num = int(name_split[3])
        if( set_num > last_set_num):
          last_set_num = set_num
    info = {
      "team1": team1,
      "team2": team2,
      "game_set": last_set_num
    }
    game_info.append(info)
  return game_info

#print(get_game_info('../../../tj/2020_spring_highlight_video/','20200305_AF_T1_SB_GEN.mp4'))

#---------------------------
# Cut Video
#---------------------------

def cutVideo(filename, inference_path, raw_path, result_path):
    result_file = open(f"{inference_path}{filename}.txt","r")
    game_date = filename.split('_')[0]
    highlight_path = findHighlightVideoPath(game_date, '/home/lol/tj/')
    print(highlight_path)
    game_info = get_game_info(highlight_path, filename + '.mp4')
    print(game_info)

    # Make pairs
    cut_ranges = []
    while True:
        start_line = result_file.readline()
        end_line = result_file.readline()
        if not start_line or not end_line : break
        
        start_split =  start_line.split()
        end_split = end_line.split()
        if start_split[0] != "start" or end_split[0] != "finish": break

        game_range = { 'start': int(start_split[2]), 'end': int(end_split[2]) }
        cut_ranges.append(game_range)

    info_index = 0
    game_index = 1
    for game_range in cut_ranges:
        print(game_range)
        if (game_info[info_index]['game_set'] < game_index):
            game_index = 1
            info_index += 1
        print(f"{result_path}{game_date}_{game_info[info_index]['team1']}_{game_info[info_index]['team2']}_{game_index}_full.mp4")
        ffmpeg_extract_subclip(
                raw_path+filename+".mp4",
                game_range['start'],
                game_range['end'],
                targetname=f"{result_path}{game_date}_{game_info[info_index]['team1'].upper()}_{game_info[info_index]['team2'].upper()}_{game_index}_full.mp4"
        )
        game_index += 1

    #ffmpeg_extract_subclip("video1.mp4", start_time, end_time, targetname="test.mpt4")
cutVideo(
    '20200228_APK_DRX_T1_SB',
    '/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/inference_result/',
    '/home/lol/lol_highlight_ai/preprocessing/downloader/full_raw/',
    '/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/full_video/'
    )

#------------------------------
# remove unnecessary timestamp
#------------------------------

def isDiffLess(a, b, min):
    print("[diff]"+str(b - a)+" [min]"+str(min*60))
    return b - a < min*60

def postprocess_timestamp(filename):
    file_in = open(filename + '_raw.txt', 'r')
    file_out = open(filename + '.txt', 'w+')

    last_end_sec = last_2_end_sec = -1
    last_end_line = last_2_end_line = ''

    last_start_sec = last_2_start_sec = -1
    last_start_line = last_2_start_line = ''
    skipEnd = False
    skipWrite = False
    rolledBack = False

    while True:
        line = file_in.readline()
        if not line: break

        line_raw = line
        line = line.strip('\n')
        line = line.split(' ')
        if(line[0] == 'start'):
            # First start
            if(last_start_sec == -1):
                last_2_start_line = last_start_line
                last_start_line = line_raw
                last_2_start_sec = last_2_start_sec
                last_start_sec = int(line[2])
                continue
            # Check if game started too quickly
            if(last_end_sec != -1 or rolledBack):
                if(isDiffLess(last_end_sec, int(line[2]), 2)):
                    skipEnd = True
                    continue
                # Normal Case
                else:
                    last_2_start_line = last_start_line
                    last_start_line = line_raw
                    last_2_start_sec = last_start_sec
                    last_start_sec = int(line[2])
            # Write previous end
            if(last_end_sec != -1 and not skipWrite):
                file_out.write(last_end_line)
            skipWrite = False
        else:
            # Only Update
            if(skipEnd):
                skipEnd = False
                last_2_end_line = last_end_line
                last_end_line = line_raw
                last_2_end_sec = last_end_sec
                last_end_sec = int(line[2])
                continue
            if(last_start_sec != -1):
                if(isDiffLess(last_start_sec, int(line[2]), 10)):
                    # Move to previous pair
                    last_start_sec = last_2_start_sec
                    last_start_line = last_2_start_line
                    last_end_sec = last_2_end_sec
                    last_end_line = last_2_end_line
                    skipWrite = True
                    rolledBack = True
                # Write previous start
                else:
                    file_out.write(last_start_line)
                    last_2_end_line = last_end_line
                    last_end_line = line_raw
                    last_2_end_sec = last_end_sec
                    last_end_sec = int(line[2])
    file_in.close()
    file_out.write(last_end_line)
    file_out.close()
#postprocess_timestamp('./inference_result/20200228_APK_DRX_T1_SB')


