from os import walk
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

HIGHLIGHT_VIDEO_PATH = '/home/lol/total_highlight_video/'

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

#folder_list = ['2019_wc_highlight_video/',
#'2020_promotion_highlight_video/','2020_spring_highlight_video/',
#'2020_msc_highlight_video/', '2020_summer_highlight_video/']
folder_list=['']
#-----------------------
# Find hightlight video from folders
#-----------------------
def findHighlightVideoPath(video_name, highlight_path):
    for folder_name in folder_list:
        result = glob.glob(f"{highlight_path}{folder_name}{video_name}*")
        if (len(result) != 0):
            return f"{highlight_path}{folder_name}"
    return ""

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
  game_info = []
  for i in range(1, int(len(file_name_split)/2)+1):
    team1 = file_name_split[i*2 - 1]
    team2 = file_name_split[i*2]
    # Get list of highlight videos
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

    # Find game from list
    for name in filtered_list:
      if (name == full_name):
        break
      else:
        # Save the biggest game set number
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

#---------------------------
# Cut Video
#---------------------------

def cutVideo(filename, inference_path, raw_path, result_path):
    print("Cut Video "+filename+" start")
    result_file = open(f"{inference_path}{filename}.txt","r")
    game_date = filename.split('_')[0]
    highlight_path = findHighlightVideoPath(game_date, HIGHLIGHT_VIDEO_PATH)
    if(highlight_path == ""):
        return

    game_info = get_game_info(highlight_path, filename + '.mp4')
    if(not game_info or len(game_info) == 0):
        print("Cut video failed: couldn't find highlight")
        exception_file = open("exceptions.txt","a")
        exception_file.write(f"{filename}\n")
        exception_file.close()
        result_file.close()
        return

    # Make start time and end time pairs
    cut_ranges = []
    while True:
        start_line = result_file.readline()
        end_line = result_file.readline()
        if not start_line or not end_line : break

        start_split =  start_line.split()
        end_split = end_line.split()
        if start_split[0] != "start" or end_split[0] != "finish": break

        game_range = { 'start': int(start_split[2]), 'end':
                int(end_split[2])+180 }
        cut_ranges.append(game_range)

    # if total number of games doesn't match number of highlight videos
    total_set = 0
    for game in game_info:
        total_set += game['game_set']
    if(len(cut_ranges) != total_set):
        print("Cut Video failed: game set doens't match")
        exception_file = open("exceptions.txt","a")
        exception_file.write(f"{filename}\n")
        exception_file.close()
        result_file.close()
        return

    # cut video
    info_index = 0
    game_index = 1
    for game_range in cut_ranges:
        if (game_info[info_index]['game_set'] < game_index):
            game_index = 1
            info_index += 1
        ffmpeg_extract_subclip(
                raw_path+filename+".mp4",
                game_range['start'],
                game_range['end'],
                targetname=f"{result_path}{game_date}_{game_info[info_index]['team1'].upper()}_{game_info[info_index]['team2'].upper()}_{game_index}_full.mp4"
        )
        game_index += 1
    print("Cut Video done")
    result_file.close()


#------------------------------
# remove unnecessary timestamp
#------------------------------

def isDiffLess(a, b, min):
    return abs(int(b) - int(a)) < min*60

MINIMUM_RUN_TIME = 8
MINIMUM_INTERVAL_TIME = 5
def postprocess_timestamp(filename):
    print("Postprocessing "+filename+" start")
    file_in = open(filename + '_raw.txt', 'r')
    file_out = open(filename + '.txt', 'w+')

    last_s_raw = last_e_raw = '' # Full line (string)
    last_s = last_e = None # Just seconds (int)

    while True:
        # EOF | Line Format: start/finish [frame] [seconds] [hr:min:sec]
        line_s_raw = file_in.readline()
        line_e_raw = file_in.readline()
        if(not line_s_raw or not line_e_raw): break

        # If out of format
        line_s = line_s_raw.strip('\n').split(' ')
        line_e = line_e_raw.strip('\n').split(' ')
        if(line_s[0] != 'start' or line_e[0] != 'finish'): break

        # First iteration
        if(last_s_raw == '' and last_e_raw == ''):
            # First section ended too quickly
            if(isDiffLess(line_s[2], line_e[2], MINIMUM_RUN_TIME)):
                continue
            last_s_raw = line_s_raw
            last_e_raw = line_e_raw
            last_s = line_s[2]
            last_e = line_e[2]
            continue

        # Game ended too quickly: 13min
        if(isDiffLess(line_s[2], line_e[2], MINIMUM_RUN_TIME)):
            continue
        # Game started too quickly: 5min
        elif(isDiffLess(last_e, line_s[2], MINIMUM_INTERVAL_TIME)):
            last_e_raw = line_e_raw
            last_e = line_e[2]
        # Normal Case
        else:
            file_out.write(last_s_raw)
            file_out.write(last_e_raw)
            last_s_raw = line_s_raw
            last_e_raw = line_e_raw
            last_s = line_s[2]
            last_e = line_e[2]
    if(last_s_raw and last_e_raw):
        file_out.write(last_s_raw)
        file_out.write(last_e_raw)
    file_in.close()
    file_out.close()
    print("Postprocessing done")



