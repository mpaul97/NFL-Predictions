import pandas as pd
import numpy as np
import urllib.request
import os
import time

DATA_PATH = "../data/"
SCRAPE_PATH = "scrapeStarters/"

def getContent(url):

    fp = urllib.request.urlopen(url)
    mybytes = fp.read()

    mystr = mybytes.decode("utf8", errors='ignore')
    fp.close()

    home_pids, home_poses, away_pids, away_poses = [], [], [], []

    # home starters
    # get table all
    temp = mystr[mystr.index('div_home_starters'):]
    table = temp[:temp.index('</table>')]

    # home_title
    home_title = table[table.index("<caption>"):table.index("</caption>")].replace("<caption>","").replace("Starters Table", "").replace(" ", "")
    
    # home_players
    home_rows = table[table.index("<tbody>"):].split("<tr >")
    for row in home_rows:
        if "<tbody>" not in row:
            if 'divider' not in row:
                # pid
                pid = row[row.index('data-append-csv=')+len('data-append-csv='):row.index('data-stat')].replace('"','').replace(" ",'')
                home_pids.append(pid)
                # position
                t0 = row.split('data-stat="pos"')[1]
                t0 = t0[2:t0.index("<")]
                home_poses.append(t0)
            else:
                temp1 = row.split("</tr>")
                temp1.pop()
                for index, row1 in enumerate(temp1):
                    # pid
                    pid = row1[row1.index('data-append-csv=')+len('data-append-csv='):row1.index('data-stat')].replace('"','').replace(" ",'')
                    home_pids.append(pid)
                    # position
                    if index == 0:
                        t0 = row.split('data-stat="pos"')[1]
                        t0 = t0[2:t0.index("<")]
                        home_poses.append(t0)
                    else:
                        t0 = row.split('data-stat="pos"')
                        t0 = t0[len(t0)-1]
                        t0 = t0[2:t0.index("<")]
                        home_poses.append(t0)

    # away
    # get table all
    temp = mystr[mystr.index('div_vis_starters'):]
    table = temp[:temp.index('</table>')]

    # home_title
    away_title = table[table.index("<caption>"):table.index("</caption>")].replace("<caption>","").replace("Starters Table", "").replace(" ", "")
    
    # home_players
    away_rows = table[table.index("<tbody>"):].split("<tr >")
    for row in away_rows:
        if "<tbody>" not in row:
            if 'divider' not in row:
                # pid
                pid = row[row.index('data-append-csv=')+len('data-append-csv='):row.index('data-stat')].replace('"','').replace(" ",'')
                away_pids.append(pid)
                # position
                t0 = row.split('data-stat="pos"')[1]
                t0 = t0[2:t0.index("<")]
                away_poses.append(t0)
            else:
                temp1 = row.split("</tr>")
                temp1.pop()
                for index, row1 in enumerate(temp1):
                    # pid
                    pid = row1[row1.index('data-append-csv=')+len('data-append-csv='):row1.index('data-stat')].replace('"','').replace(" ",'')
                    away_pids.append(pid)
                    # position
                    if index == 0:
                        t0 = row.split('data-stat="pos"')[1]
                        t0 = t0[2:t0.index("<")]
                        away_poses.append(t0)
                    else:
                        t0 = row.split('data-stat="pos"')
                        t0 = t0[len(t0)-1]
                        t0 = t0[2:t0.index("<")]
                        away_poses.append(t0)

    return home_pids, home_poses, away_pids, away_poses

def build():
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    _dir = 'scrapeStarters/'
    fns = os.listdir(_dir)
    keys = list(set([fn.replace('.csv', '').split("-")[1] for fn in fns]))
    for index, row in cd.iterrows():
        key = row['key']
        if key not in keys:
            home_abbr = row['home_abbr']
            away_abbr = row['away_abbr']
            url = "https://www.pro-football-reference.com/boxscores/" + key + ".htm"
            home_pids, home_poses, away_pids, away_poses = getContent(url)
            print(key)
            # home df
            df0 = pd.DataFrame()
            df0['starters'] = home_pids
            df0['positions'] = home_poses
            df0.to_csv("%s.csv" % (SCRAPE_PATH + home_abbr + "-" + key), index=False)
            # away df
            df1 = pd.DataFrame()
            df1['starters'] = away_pids
            df1['positions'] = away_poses
            df1.to_csv("%s.csv" % (SCRAPE_PATH + away_abbr + "-" + key), index=False)
            time.sleep(2)
            
    return

def deleteOldStarters():
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    _dir = 'scrapeStarters/'
    fns = os.listdir(_dir)
    keys = list(set([fn.replace('.csv', '').split("-")[1] for fn in fns]))
    for key in keys:
        if key not in cd['key'].values:
            temp_fns = [fn for fn in fns if key in fn]
            for fn in temp_fns:
                os.remove(_dir + fn)
    return

#############################

# deleteOldStarters()

build()