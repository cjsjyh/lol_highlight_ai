import os
team_list=['DRX','DWG','GEN','T1','AF','DYN','KT','SB','SP','HLE','GRF','APK','SRB','FPX','TES','JDG','IG','SKT','TL','AHQ','FNC','RNG','CG','G2','HKA','C9','SPY','JT','GAM','FLA','RYL','LK','FLE','MG','UOL','MMM','DFM','RY','ISG']
#you need to add team name in the league to download
sym={'다이나믹스':'DYN','아프리카':'AF','kt':'KT','담원':'DWG','설해원':'SP','샌드박스':'SB','한화생명':'HLE','그리핀':'GRF','서라벌':'SRB'}
#add synom
from stat import ST_CTIME
from pathlib import Path
import youtube_dl


youtube_playlist_link='https://www.youtube.com/playlist?list=PLIWtfvmBcNoehOmw4UANUe_XXjA3q7vRB'
#write play list link not a video's link

download_opts = { 'format' : '18', 'ignoreerrors' : True, 'playlistreverse' :
        True, 'download_archive' : '2020_spring_download_list.txt','outtmpl' :
        '%(playlist_index)s. %(title)s.%(ext)s'} 
#format can be confirmed by youtube-dl -F [url]
#current format 18 means 360p with audio

with youtube_dl.YoutubeDL(download_opts) as ydl:
    ydl.download([youtube_playlist_link])



#from this part, it sorts videos with date,team1,team2,round

for original_name in sorted(os.listdir()):
    original_name = str(original_name)
    #-----------------------------------------------------------------------
    #if '2019' in original_name.split('_')[0]:
    #    continue
    #---------------------This one for 2019 world champs--------------------
    print(original_name)
    format = original_name.split('.')[-1]
    if format != 'mp4':
        continue
    date = original_name.split('H_L ')[1][:5]
    date = '2020'+date[:2]+date[3:5]
    #-----------------------------------------------------------------------
    #if 'Play' in original_name:
    #    stage = 'p'
    #    stage += original_name[original_name.find('day')+3]
    #elif 'GROUP' in original_name:
    #    stage = 'g'
    #    stage += original_name[original_name.find('Day ')+4]
    #else:
    #    continue
    #--------------------This one for 2019 world champs ---------------------
    front_string = original_name.split('vs')[0]
    rear_string = original_name.split('vs')[1]
    for team in team_list:
        if team in front_string:
            team1 = team
        if team in rear_string:
            team2 = team
    for kor in sym:
        if kor in front_string:
            team1 = sym[kor]
        if kor in rear_string:
            team2 = sym[kor]

    print(team1,team2)    
    count = 1
    full_name = date+'_'+team1+'_'+team2+'_'+str(count)+'_'+'highlight.mp4'
    #os.rename(original_name,full_name)
    #---------------------------------------------------------------
    while True:
        if os.path.exists(full_name):
            count += 1
            full_name =date+'_'+team1+'_'+team2+'_'+str(count)+'_'+'highlight.mp4'
        else:
            os.rename(original_name,full_name)
            break
        
    #----------------------------comment here if you download 2019 world champs---------

    #in case of error from naming : (delete the error causing video and rerun with youtude-dl part commented
    #probably error making video is not real game highlight
