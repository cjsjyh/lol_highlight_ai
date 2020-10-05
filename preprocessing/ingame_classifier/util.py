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

#---------------------------
# Cut Video
#---------------------------

def cutVideo(filename, inference_path, raw_path, result_path):
    print("Cut Video "+filename+" start")
    result_file = open(f"{inference_path}{filename}.txt","r")
    game_date = filename.split('_')[0]
    highlight_path = findHighlightVideoPath(game_date, '/home/lol/tj/')
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

    info_index = 0
    game_index = 1
    for game_range in cut_ranges:
        if (game_info[info_index]['game_set'] < game_index):
            game_index = 1
            info_index += 1
        ffmpeg_extract_subclip(
                raw_path+filename+".mp4",
                game_range['start'],
                game_range['end']+180,
                targetname=f"{result_path}{game_date}_{game_info[info_index]['team1'].upper()}_{game_info[info_index]['team2'].upper()}_{game_index}_full.mp4"
        )
        game_index += 1
    print("Cut Video done")
    result_file.close()
#cutVideo(
#    '20200228_APK_DRX_T1_SB',
#    '/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/inference_result/',
#    '/home/lol/lol_highlight_ai/preprocessing/downloader/full_raw/',
#    '/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/full_video/'
#    )

a = '/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/inference_result/'
b = '/home/lol/lol_highlight_ai/preprocessing/downloader/full_raw/'
c = '/home/lol/lol_highlight_ai/preprocessing/ingame_classifier/full_video/'
cutVideo('20200325_GRF_GEN_KT_DWG_HLE_DRX',a,b,c)
cutVideo('20200326_AF_SB_T1_APK_GEN_DWG',a,b,c)
cutVideo('20200305_SB_GEN_AF_T1',a,b,c)
cutVideo('20200327_HLE_KT_SB_GRF_T1_DRX',a,b,c)
cutVideo('20200617_SB_AF_DRX_T1',a,b,c)
cutVideo('20200618_KT_DYN_HLE_SP',a,b,c)
cutVideo('20200620_DYN_AF_T1_HLE',a,b,c)
cutVideo('20200624_DYN_SP_HLE_KT',a,b,c)
cutVideo('20200625_DRX_SB_GEN_DWG',a,b,c)
cutVideo('20200626_AF_T1_HLE_DYN',a,b,c)
cutVideo('20200627_SP_GEN_DWG_KT',a,b,c)
cutVideo('20200628_AF_DRX_SB_T1',a,b,c)
cutVideo('20200402_AF_APK_DWG_HLE_KT_DRX',a,b,c)
cutVideo('20200404_APK_KT_SB_HLE_GEN_DRX',a,b,c)
cutVideo('20200405_DWG_T1_APK_GRF_HLE_AF',a,b,c)
cutVideo('20200408_DWG_GRF_APK_GEN',a,b,c)
cutVideo('20200409_SB_T1_AF_GRF',a,b,c)
cutVideo('20200207_HLE_T1_DRX_KT',a,b,c)
cutVideo('20200209_SB_DRX_AF_HLE',a,b,c)
cutVideo('20200209_SB_DRX_AF_HLE',a,b,c)
cutVideo('20200213_KT_T1_DRX_HLE',a,b,c)
cutVideo('20200215_AF_DRX_HLE_SB',a,b,c)
cutVideo('20200220_GRF_DWG_KT_APK',a,b,c)
cutVideo('20200220_GRF_DWG_KT_APK',a,b,c)
cutVideo('20200221_SB_AF_DRX_GEN',a,b,c)
cutVideo('20200222_HLE_GRF_APK_T1',a,b,c)
cutVideo('20200223_AF_KT_SB_DWG',a,b,c)
cutVideo('20200226_KT_SB_T1_GRF',a,b,c)
cutVideo('20200228_APK_DRX_T1_SB',a,b,c)
cutVideo('20200701_T1_DWG_GEN_SB',a,b,c)
cutVideo('20200702_HLE_AF_DRX_DYN',a,b,c)
cutVideo('20200416_GRF_HLE_DRX_APK',a,b,c)
cutVideo('20200420_DRX_DWG',a,b,c)
cutVideo('20200422_T1_DRX',a,b,c)
cutVideo('20200425_GEN_T1',a,b,c)
cutVideo('20200428_SB_DYN_GRF_SRB',a,b,c)

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

#postprocess_timestamp('./inference_result/20191026_GRF_IG_FPX_FNC')


