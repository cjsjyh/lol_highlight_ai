import os
team_list=['DRX','DWG','GEN','T1','AF','DYN','KT','SB','SP','HLE','GRF','APK','SRB','FPX','TES','JDG','IG','SKT','TL','AHQ','FNC','RNG','CG','G2','HKA','C9','SPY','JT','GAM','FLA','RYL','LK','FLE','MG','UOL','MMM','DFM','RY','ISG']
sym={'다이나믹스':'DYN','아프리카':'AF','담원':'DWG','설해원':'SP','샌드박스':'SB','한화생명':'HLE','그리핀':'GRF','서라벌':'SRB'}
from stat import ST_CTIME
from pathlib import Path
import youtube_dl

"""
youtube_playlist_link='https://www.youtube.com/playlist?list=PLIWtfvmBcNocAp-r377NGWt3hfxwoSWsF'
download_opts = { 'format' : '135', 'ignoreerrors' : True, 'playlistreverse' :
        True, 'download_archive' : '2019_wc_download_list.txt','outtmpl' :
        '%(playlist_index)s. %(title)s.%(ext)s'} 
with youtube_dl.YoutubeDL(download_opts) as ydl:
    ydl.download([youtube_playlist_link])
"""
for original_name in sorted(os.listdir()):
    original_name = str(original_name)
    if '2019' in original_name.split('_')[0]:
        continue
    print(original_name)
    format = original_name.split('.')[-1]
    if format != 'mp4':
        continue
    date = original_name.split('H_L ')[1][:5]
    date = '2019'+date[:2]+date[3:5]
    if 'Play' in original_name:
        stage = 'p'
        stage += original_name[original_name.find('day')+3]
    elif 'GROUP' in original_name:
        stage = 'g'
        stage += original_name[original_name.find('Day ')+4]
    else:
        continue
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

    
    count = 1
    full_name = date+'_'+team1+'_'+team2+'_'+stage+'_'+'highlight.mp4'
    os.rename(original_name,full_name)
    """
    while True:
        if os.path.exists(full_name):
            count += 1
            full_name =date+'_'+team1+'_'+team2+'_'+str(count)+'_'+'highlight.mp4'
        else:
            os.rename(original_name,full_name)
            break"""
        
