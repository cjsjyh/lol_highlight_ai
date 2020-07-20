from os import walk
import glob

# Get file names in directory
def get_filenames(path):
  f = []
  for (dirpath, dirnames, filenames) in walk(path):
      f.extend(filenames)
      break
  return f

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
    print(game_date, team1, team2)
    # Find highlight videos
    filtered_list = glob.glob(f"{highlight_path}{game_date}_{team1}_{team2}*")
    if (len(filtered_list) == 0):
        # Try with reverse order
        filtered_list = glob.glob(f"{highlight_path}{game_date}_{team2}_{team1}*")
        if(len(filtered_list) == 0):
            return None
    last_set_num = 0
    for name in filtered_list:
      if (name == full_name):
        break
      else:
        name = name.split('/')[-1]
        name_split = name.split('_')
        print(name)
        print(name_split)
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


print(get_game_info('../../../tj/2020_spring_highlight_video/','20200305_AF_T1_SB_GEN.mp4'))
