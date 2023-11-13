import pandas as pd
import numpy as np
import os
import regex as re
import urllib.request
import requests
from bs4 import BeautifulSoup

pd.options.mode.chained_assignment = None

position_sizes = {
    'QB': 1, 'RB': 2, 'WR': 4,
    'TE': 1, 'OL': 5, 'DL': 4,
    'LB': 4, 'DB': 4
}

position_attributes = [
    ('QB', 'attempted_passes'), ('RB', 'rush_attempts'), ('WR', 'receptions'),
    ('TE', 'receptions'), ('OL', ''), ('DL', 'combined_tackles'), ('LB', 'combined_tackles'),
    ('DB', 'combined_tackles')
]

MAX_POSITION_SIZES = {
    'QB': 1, 'RB': 2, 'WR': 3,
    'TE': 2, 'OL': 5, 'DL': 5,
    'LB': 3, 'DB': 6
}

def removeSamePlayers(other, filtered):
    other = list(other)
    for f in filtered:
        if f in other:
            other.remove(f)
    return other

def buildStarters(week, year):
    
    ranks = pd.read_csv("%s.csv" % ("finalSeasonRanks"))
    starters = pd.read_csv("%s.csv" % ("starters/starters_w" + str(week)))
    
    positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
    
    prevWy = str(week-1) + " | " + str(year)
    
    new_df = pd.DataFrame(columns=['key', 'wy']+positions)
    
    for index, row in starters.iterrows():
        abbr = row['key'].split("-")[0]
        players = row['players'].split("|")
        allPlayers = []
        for pos in positions:
            filteredPlayers = [p.replace(":"+pos, "") for p in players if pos in p]
            size = position_sizes[pos]
            attribute = [s[1] for s in position_attributes if s[0]==pos][0]
            if len(filteredPlayers) < size:
                if pos != 'LB' and pos != 'DL':
                    df = pd.read_csv("%s.csv" % ("positionData/" + pos + "Data"))
                else:
                    df = pd.read_csv("%s.csv" % ("positionData/LBDLData"))
                stats = df.loc[(df['wy']==prevWy)&(df['abbr']==abbr)]
                stats.sort_values(by=[attribute], ascending=False, inplace=True)
                otherPlayers = stats['p_id'].values
                otherPlayers = removeSamePlayers(otherPlayers, filteredPlayers)
                dif = size - len(filteredPlayers)
                newPlayers = "|".join(filteredPlayers + otherPlayers[:dif])
            else:
                newPlayers = "|".join(filteredPlayers)
            allPlayers.append(newPlayers)
        new_df.loc[len(new_df.index)] = [row['key'], row['wy']] + allPlayers
            
    new_df.to_csv("%s.csv" % ("starters/simpleStarters_w" + str(week)), index=False)

def convertStarters(week, year):
    
    _dir = 'starters_' + str(year)[-2:] + "/"

    df = pd.read_csv("%s.csv" % (_dir + "simpleStarters_w" + str(week)))

    positions = [(key.lower() + 's') for key in MAX_POSITION_SIZES]

    new_df = pd.DataFrame(columns=['key', 'wy', 'abbr', 'starters'])

    for index, row in df.iterrows():
        players = []
        for pos in positions:
            [players.append(p+":"+(pos.replace('s','').upper())) for p in row[pos].split("|")]
        new_df.loc[len(new_df.index)] = [row['key'], row['wy'], row['abbr'], "|".join(players)]

    new_df.to_csv("%s.csv" % (_dir + "starters_w" + str(week)), index=False)

def cleanStarters():
    
    _dir = 'starters/'
    
    for fn in os.listdir(_dir):
        if re.search(r"starters_w", fn):
            df = pd.read_csv(_dir + fn)
            new_df = pd.DataFrame(columns=['key', 'abbr', 'wy', 'starters'])
            for index, row in df.iterrows():
                key = row['key'].split("-")[1]
                abbr = row['key'].split("-")[0]
                new_df.loc[len(new_df.index)] = [key, abbr, row['wy'], row['players']]
            new_df.to_csv((_dir + fn), index=False)
    
    return

########################
# new stuff

def getNewSource(week, year):
    
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/week_' + str(week) + '.htm'
    
    wy = str(week) + " | " + str(year)

    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    links = []

    # find links
    for link in soup.find_all('a'):
        l = link.get('href')
        if (re.search(r"boxscores\/[0-9]{9}", l) or re.search(r"teams\/[a-z]{3}", l)):
            links.append(l)
            
    if 'teams' in links[0] and 'teams' in links[1]:
        links.pop(0)

    df = pd.DataFrame(columns=['key', 'abbr'])

    # parse links   
    for i in range(0, len(links)-2, 3):
        away_abbr = links[i].split("/")[2].upper()
        home_abbr = links[i+2].split("/")[2].upper()
        key = links[i+1].split("/")[2].replace(".htm","")
        if re.search(r"[0-9]{9}[a-z]{3}", key):
            df.loc[len(df.index)] = [key, home_abbr]
            df.loc[len(df.index)] = [key, away_abbr]
    
    df.insert(1, 'wy', wy)
    
    return df

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

    return list(zip(home_pids, home_poses)), list(zip(away_pids, away_poses))

def buildSimpleStarters(week, year):
    _dir = 'starters_' + str(year)[-2:] + '/'
    cols = [(key.lower() + 's') for key in MAX_POSITION_SIZES]
    new_df = pd.DataFrame(columns=['abbr']+cols)
    df = pd.read_csv("%s.csv" % "gameData")
    source = getNewSource(week, year)
    if week == 1 and year == 2023:
        sdf = pd.read_csv("%s.csv" % "starters_23/cleanStarters_w1")
        for index, row in sdf.iterrows():
            abbr = row['abbr']
            starters = (row['starters']).split("|")
            starters = np.array([(s.split(":")) for s in starters])
            all_pids = []
            for key in MAX_POSITION_SIZES:
                pids = [s[0] for s in starters if s[1] in key]
                pids = '|'.join(pids)
                all_pids.append(pids)
            new_df.loc[len(new_df.index)] = [abbr] + all_pids
        source = source.merge(new_df, on=['abbr'])
        source.to_csv("%s.csv" % (_dir + "simpleStarters_w1"), index=False)
    return

############################

convert = 1
week = 1
year = 2023

if convert == 0:
    # buildStarters(
    #     week=week,
    #     year=year
    # )
    # print("Update simple starters then convert...")
    buildSimpleStarters(week, year)
else:
    convertStarters(
        week=week,
        year=year
    )