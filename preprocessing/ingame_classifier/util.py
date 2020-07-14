from os import walk
import glob

# Get file names in directory
def get_filenames(path):
  f = []
  for (dirpath, dirnames, filenames) in walk(path):
      f.extend(filenames)
      break
  return f

def get_game_info(full_name):
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
    
    filtered_list = glob.glob(f"{game_date}_{team1}_{team2}*")
    last_set_num = 0
    for name in filtered_list:
      if (name == full_name):
        break
      else:
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


print(get_game_info('20200628_AF_DRX_SKT_GEN.mp4'))
