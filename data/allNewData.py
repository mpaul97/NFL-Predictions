import pandas as pd
import numpy as np
import os
import urllib.request
import time
import sys
import requests
import regex as re
from bs4 import BeautifulSoup
sys.path.append("../")
from starters.startsInfo import StartsInfo
from snapCounts.main import Main as SCMain
from ordered_set import OrderedSet
import math

from sportsipy.nfl.boxscore import Boxscores
from sportsipy.nfl.boxscore import Boxscore
from sportsipy.nfl.teams import Teams
from sportsipy.nfl.roster import Player

from fantasyData import concatFantasyAndSkillData
from approximateValues import ApproximateValues
from olStatsData import OlStatsData

NAMES_PATH = "../playerNames/"
STARTERS_PATH = "../starters/"

PERC_COLS = {
    'QB': 'attempted_passes', 'RB': 'total_touches',
    'WR': 'times_pass_target', 'TE': 'times_pass_target'
}

SEASON_POSITIONS = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'K', 'P', 'LS']

position_sizes = [
    ('QB', 1),
    ('RB', 2),
    ('WR', 4),
    ('TE', 1),
    ('OL', 5),
    ('DL', 4),
    ('LB', 4),
    ('DB', 4)
]

position_attributes = [
    ('QB', 'attempted_passes'),
    ('RB', 'rush_attempts'),
    ('WR', 'receptions'),
    ('TE', 'receptions'),
    ('OL', ''),
    ('DL', 'combined_tackles'),
    ('LB', 'combined_tackles'),
    ('DB', 'combined_tackles')
]

def buildRawData(week, year, _dir):

    box_list, df_list = [], []
    box_list.append(Boxscores(week, year))

    for boxscores in box_list:
        for key in boxscores.games.keys():
            for item in boxscores.games[key]:
                l1 = list(item.values())
                boxscore = Boxscore(l1[0])
                try:
                    temp_df = boxscore.dataframe
                    temp_df['winning_abbr'] = boxscore.winning_abbr
                    temp_df['home_abbr'] = boxscore.home_abbreviation.upper()
                    temp_df['away_abbr'] = boxscore.away_abbreviation.upper()
                    df_list.append(temp_df)
                except TypeError:
                    print('type error:', l1[0])
                time.sleep(0.5)

    #raw stats
    df = pd.concat(df_list)
    df['wy'] = [(str(week) + " | " + str(year)) for _ in range(len(df.index))]
    df.insert(0, 'key', df.index)
    
    df.to_csv("%s.csv" % (_dir + "rawData_" + str(week) + "-" + str(year)), index=False)
    
    return

def buildRawData_fixed(week, year, _dir):
    wrong_cols = [
        'losing_abbr', 'losing_name', 'winner',
        'winning_abbr', 'winning_name', 'home_points',
        'away_points'
    ]
    box_list, df_list = [], []
    box_list.append(Boxscores(week, year))
    for boxscores in box_list:
        for key in boxscores.games.keys():
            for item in boxscores.games[key]:
                l1 = list(item.values())
                boxscore = Boxscore(l1[0])
                try:
                    temp_df = boxscore.dataframe
                    temp_df['winning_abbr'] = boxscore.losing_abbr
                    temp_df['losing_abbr'] = boxscore.winning_abbr
                    temp_df['winning_name'] = boxscore.losing_name
                    temp_df['losing_name'] = boxscore.winning_name
                    temp_df['winner'] = 'Away' if boxscore.winner == 'Home' else 'Home'
                    temp_df['home_points'] = boxscore.away_points
                    temp_df['away_points'] = boxscore.home_points
                    temp_df['home_abbr'] = boxscore.home_abbreviation.upper()
                    temp_df['away_abbr'] = boxscore.away_abbreviation.upper()
                    df_list.append(temp_df)
                except TypeError:
                    print('type error:', l1[0])
    #raw stats
    df = pd.concat(df_list)
    df['wy'] = [(str(week) + " | " + str(year)) for _ in range(len(df.index))]
    df.insert(0, 'key', df.index)
    df.to_csv("%s.csv" % (_dir + "rawData_" + str(week) + "-" + str(year)), index=False)
    return

def buildPlayerData(week, year, _dir):
    box_list, all_players = [], []
    box_list.append(Boxscores(week, year))
    count = 0
    for boxscores in box_list:
        for key in boxscores.games.keys():
            for item in boxscores.games[key]:
                l1 = list(item.values())#values from boxscores to get game IDs
                boxscore = Boxscore(l1[0])
                try:
                    h_players = boxscore.home_players
                    a_players = boxscore.away_players
                    h_abbr = boxscore.home_abbreviation.upper()
                    a_abbr = boxscore.away_abbreviation.upper()
                    for p in h_players:
                        df_p = p.dataframe
                        df_p['abbr'] = h_abbr
                        # df_p['wy'] = wy[count]
                        df_p['game_key'] = l1[0]
                        df_p['isHome'] = True
                        all_players.append(df_p)
                    for p in a_players:
                        df_p = p.dataframe
                        df_p['abbr'] = a_abbr
                        # df_p['wy'] = wy[count]
                        df_p['game_key'] = l1[0]
                        df_p['isHome'] = False
                        all_players.append(df_p)
                except (AttributeError, TypeError) as error:
                    print('error:', error, l1[0])
                time.sleep(5)
        count += 1
    df_all = pd.concat(all_players)
    df_all['wy'] = [(str(week) + " | " + str(year)) for _ in range(len(df_all.index))]
    df_all.fillna(0, inplace=True)
    df_all = df_all.reset_index()
    df_all = df_all.rename(columns={'index': 'p_id'})
    df_all.to_csv('%s.csv' % (_dir + "playerData_" + str(week) + "-" + str(year)), index=False)
    return

######################################
# convert data

def convertTime(time):
    temp1 = time.split(":")
    temp = int(temp1[0])
    if temp == 1 or ("am" in temp1[1]) or temp == 12:
        return 0
    elif temp >= 2 and temp < 7:
        return 1
    elif temp >= 7:
        return 2

def timeToInt(posTime):
	if pd.isna(posTime):
		posTime = "3:00"
	temp = posTime.split(":")
	num = (int(temp[0])*60) + int(temp[1])
	return num

def convertWeather(weather):
	if not pd.isna(weather):
		wl = weather.split(", ")
		temperature = 0
		humidity = -1
		wind = 0
		for w in wl:
			if "degrees" in w:
				temperature = int(w[:2])
			elif "humidity" in w:
				humidity = int(w[-3:].replace("%", ""))
			elif "mph" in w:
				wind = int(w.split(" ")[1])
		if humidity == -1:
			humidity = 65
		return str(temperature) + "|" + str(humidity) + "|" + str(wind)
	else:
		return "Dome"

def convertSurface(surface):
	# grass = 0, turf = 1
	if isinstance(surface, float):
		return 0
	if "turf" not in surface:
		return 0
	else:
		return 1

def convertStadium(stadium):
    try:
        temp = stadium.split(" ")[0]
        sum_id = 0
        for c in temp:
            sum_id += ord(c)
        return sum_id
    except AttributeError:
        return np.NaN

def getLineHit(row, dfan):
    line = row['vegas_line']
    if not pd.isna(line):
        if "Pick" not in line:
            temp = line.split("-")
            name = temp[0].replace(" ", "")
            if name == "WashingtonRedskins" or name == 'WashingtonCommanders':
                name = "WashingtonFootballTeam"
            if name == "OaklandRaiders":
                name = "LasVegasRaiders"
            if name == "St.LouisRams":
                name = "LosAngelesRams"
            if name == "SanDiegoChargers":
                name = "LosAngelesChargers"
            if name == "HoustonOilers":
                name = "TennesseeTitans"
            if name == "St.LouisCardinals":
                name = "ArizonaCardinals"
            if name == "BaltimoreColts":
                name = "IndianapolisColts"
            if name == "LosAngelesRaiders":
                name = "LasVegasRaiders"
            if name == "PhoenixCardinals":
                name = "ArizonaCardinals"
            if name == "TennesseeOilers":
                name = "TennesseeTitans"
            linePoints = float(temp[1])
            favAbbr = dfan.loc[dfan['name']==name, 'abbr'].values[0]
            #0 no hit, 1 hit
            if favAbbr == row['home_abbr']:
                diff = row['home_points'] - row['away_points']
                if diff > linePoints:
                    return 1
                else:
                    return 0
            else:
                diff = row['away_points'] - row['home_points']
                if diff > linePoints:
                    return 1
                else:
                    return 0
        else:
            return 0
    return 0

def getMonth(datetime):
	temp = datetime.split("-")
	return temp[1]

def getOUHit(row):
	#0 no hit, 1 hit
	if not isinstance(row['over_under'], float):
		temp = row['over_under'].split("(")
		ouPoints = float(temp[0].replace(" ", ""))
		overOrUnder = temp[1].replace(")", "")
		totalPoints = row['home_points'] + row['away_points']
		if overOrUnder == "over":
			if totalPoints > ouPoints:
				return 1
			else:
				return 0
		else:
			if totalPoints < ouPoints:
				return 1
			else:
				return 0
	else:
		return 0

def convertRawData(week, year, _dir):
    
    df = pd.read_csv("%s.csv" % (_dir + "rawData_" + str(week) + "-" + str(year)))
    #convert data
    dfan = pd.read_csv("%s.csv" % "abbrWithNames")

    #time of game
    df['time'] = [convertTime(t) for t in list(df['time'])]

    #weather
    df['weather'] = [convertWeather(w) for w in list(df['weather'])]

    #surface
    df['surface'] = [convertSurface(s) for s in list(df['surface'])]

    #stadium
    df['stadium_id'] = [convertStadium(s) for s in list(df['stadium'])]

    #vegas line
    lineHit = []
    for index, row in df.iterrows():
        lineHit.append(getLineHit(row, dfan))

    df['lineHit'] = lineHit

    #time of poss, duration
    df['home_time_of_possession'] = [timeToInt(t) for t in list(df['home_time_of_possession'])]
    df['away_time_of_possession'] = [timeToInt(t) for t in list(df['away_time_of_possession'])]
    df['duration'] = [timeToInt(t) for t in list(df['duration'])]

    #month
    df['month'] = [getMonth(d) for d in list(df['datetime'])]

    #overunder hit
    ouHit = []
    for index, row in df.iterrows():
        ouHit.append(getOUHit(row))

    df['ouHit'] = ouHit

    df = df.fillna(df.mean())

    df = df.round(0)

    df.to_csv("%s.csv" % (_dir + "convertedData_" + str(week) + "-" + str(year)), index=False)
    
    return

######################################

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

def buildScrapeStarters(week, year, _dir):
    
    cd = pd.read_csv("%s.csv" % (_dir + "convertedData_" + str(week) + "-" + str(year)))
    sp = pd.read_csv("%s.csv" % "simplePositions")
    
    df = pd.DataFrame(columns=['key', 'wy', 'starters'])

    for index, row in cd.iterrows():
        key = row['key']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        url = "https://www.pro-football-reference.com/boxscores/" + key + ".htm"
        home_pids, home_poses, away_pids, away_poses = getContent(url)
        home_poses = [sp.loc[sp['position']==pos, 'simplePosition'].values[0] for pos in home_poses]
        away_poses = [sp.loc[sp['position']==pos, 'simplePosition'].values[0] for pos in away_poses]
        print(key)
        wy = str(week) + " | " + str(year)
        # home
        home_key = home_abbr + "-" + key
        home_starters = "|".join([home_pids[i] + ":" + home_poses[i] for i in range(len(home_pids))])
        df.loc[len(df.index)-1] = [home_key, wy, home_starters]
        # away
        away_key = away_abbr + "-" + key
        away_starters = "|".join([away_pids[i] + ":" + away_poses[i] for i in range(len(away_pids))])
        df.loc[len(df.index)-1] = [away_key, wy, away_starters]
        time.sleep(5)
        
    df.to_csv("%s.csv" % (_dir + "starters_" + str(week) + "-" + str(year)), index=False)
        
    return

def concatStarters(week, year, _dir):
    df0 = pd.read_csv("%s.csv" % (_dir + "starters_" + str(week) + "-" + str(year)))
    df0.insert(1, 'abbr', [row['key'].split('-')[0] for _, row in df0.iterrows()])
    df0['key'] = [row['key'].split('-')[1] for _, row in df0.iterrows()]
    df1 = pd.read_csv("%s.csv" % (STARTERS_PATH + "allStarters"))
    df1 = pd.concat([df1, df0])
    df1.to_csv("%s.csv" % (STARTERS_PATH + "allStarters"), index=False)
    return

def updateStartsInfo(_dir):
    si = StartsInfo(_dir)
    si.updateStartsInfo()
    return

######################################

def getPosition(pid):

    first_key = pid[0].upper()
    url = "https://www.pro-football-reference.com/players/" + first_key + "/" + pid + ".htm"

    fp = urllib.request.urlopen(url)
    mybytes = fp.read()

    mystr = mybytes.decode("utf8", errors='ignore')
    fp.close()

    try:
        temp = mystr[mystr.index('<strong>Position'):]
        temp = temp[temp.index(":"):]
        temp = temp[:5]

        pos = temp.replace(":", '').replace(" ","")

        pos = "".join(pos.split())
    except ValueError:
        print(pid)
        pos = 'UNK'

    return pos

def mostCommon(List):
    return max(set(List), key = List.index)

def getNewYears(df):
    curr_years = df['year'].values
    new_years = []
    last_valid_year = 0
    for index, year in enumerate(curr_years):
        if type(year) is float and math.isnan(year):
            if index != 0:
                new_years.append(last_valid_year)
        else:
            if (type(year) is str or type(year) is object) and '*' in year:
                year = year.replace('*', '')
            if (type(year) is str or type(year) is object) and '+' in year:
                year = year.replace('+', '')
            last_valid_year = year
            new_years.append(year)
    return new_years

def getPosition_v2(pid):
    url_c = pid[0].upper()
    url = 'https://www.pro-football-reference.com/players/' + url_c + '/' + pid + '.htm'
    tables = pd.read_html(url)
    df_list = []
    for t in tables:
        if type(t.columns[0]) is tuple or len(tables) == 1 or (type(t.columns[0]) is str and 'Year' in t.columns):
            temp_df = t[t.columns[:4]]
            temp_df.columns = ['year', 'age', 'team', 'position']
            temp_df['year'] = getNewYears(temp_df)
            temp_df['year'] = pd.to_numeric(temp_df['year'], errors='coerce')
            temp_df.dropna(subset=['year'], inplace=True)
            df_list.append(temp_df)   
    df = pd.concat(df_list)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['year'], inplace=True)
    df.drop(columns=['age'], inplace=True)
    df.dropna(inplace=True)
    position = mostCommon([pos for pos in df['position'].values if 'Missed season' not in pos])
    return position

def buildPlayerPositions(week, year, _dir):

    all_data = []
    data_dir = "positionData/"
    [all_data.append(pd.read_csv(data_dir + fn)) for fn in os.listdir(data_dir) if 'csv' in fn]
    all_df = pd.concat(all_data)
    all_df = all_df[['p_id', 'position']]

    cd = pd.read_csv("%s.csv" % (_dir + "playerData_" + str(week) + "-" + str(year)))
    spdf = pd.read_csv("%s.csv" % (NAMES_PATH + "positionsFinalPlayerInfo"))
    positions = []

    for index, row in cd.iterrows():
        print(str(round((index/len(cd.index))*100, 2)) + '%')
        pid = row['p_id']
        if pid not in all_df['p_id'].values:
            pos = getPosition_v2(pid)
            s_pos = spdf.loc[spdf['position']==pos, 'simplePosition'].values[0]
            positions.append(s_pos)
            time.sleep(5)
        else:
            position = all_df.loc[all_df['p_id']==pid, 'position'].values[0]
            positions.append(position)

    cd['position'] = positions

    cd.to_csv("%s.csv" % (_dir + "positionPlayerData_" + str(week) + "-" + str(year)), index=False)

def splitPositions(week, year, _dir):
    
    df = pd.read_csv("%s.csv" % (_dir + "positionPlayerData_" + str(week) + "-" + str(year)))
    tp = pd.read_csv("%s.csv" % (NAMES_PATH + "finalPlayerInfo"))
    sp = pd.read_csv("%s.csv" % "simplePositions")
    
    positions = list(set(list(sp['simplePosition'])))
    
    simpleList = []
    
    for index, row in df.iterrows():
        pid = row['p_id']
        position = tp.loc[tp['p_id']==pid, 'position'].values
        if len(position) > 0:
            position = position[0]
            simplePosition = sp.loc[sp['position']==position, 'simplePosition'].values[0]
            simpleList.append(simplePosition)
        else:
            simpleList.append(row['position'])
        
    df['position'] = simpleList

    if "positionData" not in os.listdir(_dir):
        os.mkdir(_dir + "positionData/")

    for pos in positions:
        temp = df.loc[df['position']==pos]
        temp.to_csv("%s.csv" % (_dir + "positionData/" + pos + "Data_" + str(week) + "-" + str(year)), index=False)

    return

######################################

def cleanQb(suffix, _dir):
    
    df = pd.read_csv("%s.csv" % (_dir + "QBData" + suffix))

    cols = list(df.columns)

    startCol = 'times_pass_target'
    endCol ='longest_punt'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % (_dir + "QBData" + suffix), index=False)

def cleanRb(suffix, _dir):

    df = pd.read_csv("%s.csv" % (_dir + "RBData" + suffix))

    cols = list(df.columns)

    startCol = 'completed_passes'
    endCol ='quarterback_rating'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    startCol1 = 'interceptions'
    endCol1 = 'longest_punt'

    startIndex1 = cols.index(startCol1)
    endIndex1 = cols.index(endCol1) + 1

    drop_cols += cols[startIndex1:endIndex1]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % (_dir + "RBData" + suffix), index=False)

def cleanWr(suffix, _dir):

    df = pd.read_csv("%s.csv" % (_dir + "WRData" + suffix))

    cols = list(df.columns)

    startCol = 'completed_passes'
    endCol ='quarterback_rating'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    startCol1 = 'interceptions'
    endCol1 = 'longest_punt'

    startIndex1 = cols.index(startCol1)
    endIndex1 = cols.index(endCol1) + 1

    drop_cols += cols[startIndex1:endIndex1]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % (_dir + "WRData" + suffix), index=False)

def cleanTe(suffix, _dir):

    df = pd.read_csv("%s.csv" % (_dir + "TEData" + suffix))

    cols = list(df.columns)

    startCol = 'completed_passes'
    endCol ='quarterback_rating'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    startCol1 = 'interceptions'
    endCol1 = 'longest_punt'

    startIndex1 = cols.index(startCol1)
    endIndex1 = cols.index(endCol1) + 1

    drop_cols += cols[startIndex1:endIndex1]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % (_dir + "TEData" + suffix), index=False)
   
def cleanPositionData(week, year, _dir):
    
    suffix = "_" + str(week) + "-" + str(year)
    
    _dir += "positionData/"
    
    cleanQb(suffix, _dir)
    cleanRb(suffix, _dir)
    cleanWr(suffix, _dir)
    cleanTe(suffix, _dir)
    
    return
    
######################################

def zeroDivision(n, d):
    	return n / d if d else 0

def buildExtraQb(suffix, _dir):

	pl = pd.read_csv("%s.csv" % (_dir + "QBData" + suffix))

	compL = []
	tdL = []
	intL = []
	ypL = []
	aypL = []
	ycL = []
	spL = []
	ngL = []
	anyL = []
	rpaL = []

	for index, row in pl.iterrows():
		# completion percentage
		comp = int(row['completed_passes'])
		att = int(row['attempted_passes'])
		compPerc = (zeroDivision(comp, att))*100
		compL.append(compPerc)
		# td percentage
		tds = int(row['passing_touchdowns'])
		tdPerc = (zeroDivision(tds, att))*100
		tdL.append(tdPerc)
		# interception percentage
		ints = int(row['interceptions_thrown'])
		intPerc = (zeroDivision(ints, att))*100
		intL.append(intPerc)
		# yards per attempt
		yards = int(row['passing_yards'])
		ypa = (zeroDivision(yards, att))
		ypL.append(ypa)
		# adjusted yards per attempt
		apa = zeroDivision((yards + 20 * tds - 45 * ints), att)
		aypL.append(apa)
		# yards per completion
		ypc = zeroDivision(yards, comp)
		ycL.append(ypc)
		# sack percentage
		tsk = int(row['times_sacked'])
		skp = zeroDivision(tsk, (att + tsk))*100
		spL.append(skp)
		# net gained per pass attempt
		sky = int(row['yards_lost_from_sacks'])
		ng = zeroDivision((yards - sky), (att + tsk))
		ngL.append(ng)
		# adjusted net yards per pass attempt
		anya = zeroDivision((yards - sky + (20 * tds) - (45 * ints)), (att + tsk))
		anyL.append(anya)
		# rush yards per attempt
		ry = int(row['rush_yards'])
		ra = int(row['rush_attempts'])
		rypa = zeroDivision(ry, ra)
		rpaL.append(rypa)

	pl['completion_percentage'] = compL
	pl['td_percentage'] = tdL
	pl['interception_percentage'] = intL
	pl['yards_per_attempt'] = ypL
	pl['adjusted_yards_per_attempt'] = aypL
	pl['yards_per_completion'] = ycL
	pl['sack_percentage'] = spL
	pl['net_gained_per_pass_attempt'] = ngL
	pl['adjusted_net_yards_per_pass_attempt'] = anyL
	pl['rush_yards_per_attempt'] = rpaL

	pl = pl.round(2)

	pl.to_csv("%s.csv" % (_dir + "QBData" + suffix), index=False)

def buildExtraRw(suffix, _dir):

    rd = pd.read_csv("%s.csv" % (_dir + "RBData" + suffix))
    wd = pd.read_csv("%s.csv" % (_dir + "WRData" + suffix))
    td = pd.read_csv("%s.csv" % (_dir + "TEData" + suffix))
    pl = pd.concat([rd, wd, td])

    rpaL = []
    rprL = []
    catchpL = []
    rptL = []
    touchL = []
    ysL = []
    sptL = []
    totalL = []
    tptL = []
    tprL = []
    tpruL = []
    tptaL = []

    for index, row in pl.iterrows():
        # rush yards per attempt
        ry = int(row['rush_yards'])
        ra = int(row['rush_attempts'])
        rypa = zeroDivision(ry, ra)
        rpaL.append(rypa)
        # receiving yards per reception
        recy = int(row['receiving_yards'])
        recs = int(row['receptions'])
        rypr = zeroDivision(recy, recs)
        rprL.append(rypr)
        # catch percentage
        target = int(row['times_pass_target'])
        cp = zeroDivision(recs, target)*100
        catchpL.append(cp)
        # receiving yards per target
        rect = zeroDivision(recy, target)
        rptL.append(rect)
        # total touches
        touch = ra + recs
        touchL.append(touch)
        # yards from scrimmage
        ys = ry + recy
        ysL.append(ys)
        # scrimmage yards per touch
        spt = zeroDivision(ys, touch)
        sptL.append(spt)
        # total touchdowns
        rut = int(row['rush_touchdowns'])
        rt = int(row['receiving_touchdowns'])
        total = rut + rt
        totalL.append(total)
        # touchdown per touch
        tpt = zeroDivision(total, touch)
        tptL.append(tpt)
        # touchdown per reception
        tpr = zeroDivision(rt, recs)
        tprL.append(tpr)
        # touchdown per rush
        tpru = zeroDivision(rut, ra)
        tpruL.append(tpru)
        # touchdown per target
        tpta = zeroDivision(rt, target)
        tptaL.append(tpta)

    pl['rush_yards_per_attempt'] = rpaL
    pl['receiving_yards_per_reception'] = rprL
    pl['catch_percentage'] = catchpL
    pl['receiving_yards_per_target'] = rptL
    pl['total_touches'] = touchL
    pl['yards_from_scrimmage'] = ysL
    pl['scrimmage_yards_per_touch'] = sptL
    pl['total_touchdowns'] = totalL
    pl['touchdown_per_touch'] = tptL
    pl['touchdown_per_reception'] = tprL
    pl['touchdown_per_rush'] = tpruL
    pl['touchdown_per_target'] = tptaL

    pl = pl.round(3)

    rbs = pl.loc[pl['position']=='RB']
    wrs = pl.loc[pl['position']=='WR']
    tes = pl.loc[pl['position']=='TE']

    rbs.to_csv("%s.csv" % (_dir + "RBData" + suffix), index=False)
    wrs.to_csv("%s.csv" % (_dir + "WRData" + suffix), index=False)
    tes.to_csv("%s.csv" % (_dir + "TEData" + suffix), index=False)

def addExtraData(week, year, _dir):
    
    suffix = "_" + str(week) + "-" + str(year)
    
    _dir += "positionData/"
    
    buildExtraQb(suffix, _dir)
    buildExtraRw(suffix, _dir)
    
    return

# add attempted_passes, rush_attempts, and targets percentages
def addPercentages(week, year, _dir):
    suffix = "_" + str(week) + "-" + str(year)
    _dir += "positionData/"
    positions = ['QB', 'RB', 'WR', 'TE']
    for pos in positions:
        fn = pos + 'Data_' + str(week) + '-' + str(year) + '.csv'
        df = pd.read_csv(_dir + fn)
        keys = list(set(df['game_key'].values))
        df_list = []
        for key in keys:
            data: pd.DataFrame = df.loc[df['game_key']==key]
            abbrs = list(set(data['abbr'].values))
            for abbr in abbrs:
                stats: pd.DataFrame = data.loc[data['abbr']==abbr]
                if stats.shape[0] > 1:
                    total = sum(stats[PERC_COLS[pos]].values)
                    stats['volume_percentage'] = stats[PERC_COLS[pos]] / total
                else:
                    stats['volume_percentage'] = 1.0
                df_list.append(stats)
        new_df = pd.concat(df_list)
        new_df = new_df.round(2)
        new_df.sort_values(by=['game_key', 'abbr'], inplace=True)
        new_df.to_csv(_dir + fn, index=False)
    return

######################################

def mergePositions(week, year, _dir, fn0, fn1):
    
    df = pd.read_csv("%s.csv" % (_dir + "positionPlayerData_" + str(week) + "-" + str(year)))
    tp = pd.read_csv("%s.csv" % (NAMES_PATH + "finalPlayerInfo"))
    sp = pd.read_csv("%s.csv" % "simplePositions")
    
    simpleList = []
    
    for index, row in df.iterrows():
        pid = row['p_id']
        position = tp.loc[tp['p_id']==pid, 'position'].values
        if len(position) > 0:
            position = position[0]
            simplePosition = sp.loc[sp['position']==position, 'simplePosition'].values[0]
            simpleList.append(simplePosition)
        else:
            simpleList.append(row['position'])
        
    df['position'] = simpleList

    df: pd.DataFrame = df.loc[(df['position']==fn0)|(df['position']==fn1)]
    
    if fn0 == 'LB' and fn1 == 'DL':
        cols = [
            'p_id', 'interceptions', 'yards_returned_from_interception',
            'interceptions_returned_for_touchdown', 'longest_interception_return',
            'passes_defended', 'sacks', 'combined_tackles', 'solo_tackles',
            'assists_on_tackles', 'tackles_for_loss', 'quarterback_hits',
            'fumbles_recovered', 'yards_recovered_from_fumble',
            'fumbles_recovered_for_touchdown', 'fumbles_forced',
            'kickoff_returns', 'kickoff_return_yards', 'average_kickoff_return_yards',
            'kickoff_return_touchdown', 'longest_kickoff_return', 'punt_returns',
            'punt_return_yards', 'yards_per_punt_return', 'punt_return_touchdown',
            'abbr', 'game_key', 'isHome', 'wy', 'position'
        ]
        df = df[cols]
    
    df.to_csv("%s.csv" % (_dir + "positionData/" + (fn0 + fn1 + "Data_" + str(week) + "-" + str(year))), index=False)

######################################

def concatFiles(week, year, _dir):
    
    suffix = "_" + str(week) + "-" + str(year)
    
    # gameData
    gd0 = pd.read_csv("%s.csv" % "gameData")
    gd1 = pd.read_csv("%s.csv" % "oldGameData_78")
    new_gd = pd.read_csv("%s.csv" % (_dir + "convertedData" + suffix))
    
    gd0 = pd.concat([gd0, new_gd])
    gd0.drop_duplicates(inplace=True)
    gd0.to_csv("%s.csv" % "gameData", index=False)
    
    gd1 = pd.concat([gd1, new_gd])
    gd1.drop_duplicates(inplace=True)
    gd1.to_csv("%s.csv" % "oldGameData_78", index=False)
    
    # playerData
    _dir += "positionData/"

    for fn in os.listdir(_dir):
        if 'csv' in fn:
            df0 = pd.read_csv(_dir + fn)
            df1 = pd.read_csv("positionData/" + fn.replace(suffix, ""))
            temp_df = pd.concat([df1, df0])
            temp_df.drop_duplicates(inplace=True)
            temp_df.to_csv("positionData/" + fn.replace(suffix, ""), index=False)

    return

def getNewSource(week, year):
    
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/week_' + str(week) + '.htm'
    
    wy = str(week) + " | " + str(year)
    
    print('Getting new source: ' + wy + ' ...')

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

    df = pd.DataFrame(columns=['key', 'wy', 'home_abbr', 'away_abbr'])

    # parse links   
    for i in range(0, len(links)-2, 3):
        away_abbr = links[i].split("/")[2].upper()
        home_abbr = links[i+2].split("/")[2].upper()
        key = links[i+1].split("/")[2].replace(".htm","")
        if re.search(r"[0-9]{9}[a-z]{3}", key):
            df.loc[len(df.index)] = [key, wy, home_abbr, away_abbr]
    
    return df

def duplicateStarters(week, year, _dir):
    df = pd.read_csv("%s.csv" % (_dir + "starters_" + str(week) + "-" + str(year)))
    df['abbr'] = df['key'].apply(lambda x: x.split("-")[0])
    df['key'] = df['key'].apply(lambda x: x.split("-")[1])
    df = df[['key', 'wy', 'abbr', 'starters']]
    source: pd.DataFrame = getNewSource((week+1), year)
    new_keys = []
    for index, row in df.iterrows():
        abbr = row['abbr']
        try:
            key = source.loc[(source['home_abbr']==abbr)|(source['away_abbr']==abbr), 'key'].values[0]
        except IndexError:
            key = row['key']
        new_keys.append(key)
    df['key'] = new_keys
    df.sort_values(by=['key'], inplace=True)
    df['wy'] = str(week+1) + " | " + str(year)
    df.to_csv("%s.csv" % ("starters_" + str(year)[-2:] + "/starters_w" + str(week+1)), index=False)
    return

######################################

def getSummaries(week, year, _dir): 
    df = pd.read_csv("%s.csv" % "summaries")
    cd = pd.read_csv("%s.csv" % (_dir + "convertedData_" + str(week) + "-" + str(year)))
    columns = [
        'key', 'wy', 'home_abbr', 'home_1', 'home_2', 'home_3', 'home_4', 'home_ot',
        'away_abbr', 'away_1', 'away_2', 'away_3', 'away_4', 'away_ot'
    ]
    new_df = pd.DataFrame(columns=columns)
    for index, row in cd.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        boxscore = Boxscore(key)
        summary = boxscore.summary
        home_sum = '|'.join([str(i) for i in summary['home']])
        away_sum = '|'.join([str(i) for i in summary['away']])
        # home
        home_sum = home_sum.split("|")
        if len(home_sum) < 5:
            home_sum.append(0)
        if len(home_sum) > 5:
            home_sum.pop(len(home_sum)-2)
        home_sum = [int(s) for s in home_sum]
        # away
        away_sum = away_sum.split("|")
        if len(away_sum) < 5:
            away_sum.append(0)
        if len(away_sum) > 5:
            away_sum.pop(len(away_sum)-2)
        away_sum = [int(s) for s in away_sum]
        new_df.loc[len(new_df.index)-1] = [
            row['key'], row['wy'], home_abbr
        ] + home_sum + [away_abbr] + away_sum
    new_df.to_csv("%s.csv" % (_dir + "summaries_" + str(week) + "-" + str(year)), index=False)
    df = pd.concat([df, new_df])
    df.to_csv("%s.csv" % "summaries", index=False)
    return

def allPids(week, year, _dir): # everyone that played for given week
    df = pd.read_csv("%s.csv" % "allPids")
    wy = str(week) + ' | ' + str(year)
    if wy not in df['wy'].values:
        sdf = pd.read_csv("%s.csv" % (_dir + "starters_" + str(week) + "-" + str(year)))
        pdf = pd.read_csv("%s.csv" % (_dir + "positionPlayerData_" + str(week) + "-" + str(year)))
        new_df = pd.DataFrame(columns=['key', 'wy', 'abbr', 'pids'])
        for index, row in sdf.iterrows():
            abbr = (row['key']).split('-')[0]
            key = (row['key']).split('-')[1]
            wy = row['wy']
            starters = [(s.split(":")[0], s.split(":")[1]) for s in (row['starters']).split('|')]
            others = pdf.loc[(pdf['abbr']==abbr)&(pdf['wy']==wy), ['p_id', 'position']].values
            others = list(zip(others[:, 0], others[:, 1]))
            dif = list(set(others).difference(set(starters)))
            all_pids = '|'.join([(s[0] + ':' + s[1]) for s in dif]) + '|' + row['starters']
            new_df.loc[len(new_df.index)] = [key, wy, abbr, all_pids]
        df = pd.concat([df, new_df])
        df.to_csv("%s.csv" % "allPids", index=False)
    else:
        print('allPids already included.')
    return

def allSeasonPlayers(week, year, _dir): # everyone that played for given year
    df = pd.read_csv("%s.csv" % "allSeasonPlayers")
    sdf = pd.read_csv("%s.csv" % (_dir + "starters_" + str(week) + "-" + str(year)))
    pdf = pd.read_csv("%s.csv" % (_dir + "positionPlayerData_" + str(week) + "-" + str(year)))
    cols = [(pos.lower() + 's') for pos in SEASON_POSITIONS]
    new_df = pd.DataFrame(columns=['year', 'abbr']+cols)
    for index, row in sdf.iterrows():
        abbr = row['key'].split("-")[0]
        all_starters = [(s.split(":")[0], s.split(":")[1]) for s in (row['starters']).split("|")]
        others = pdf.loc[pdf['abbr']==abbr, ['p_id', 'position']].values
        others = list(zip(others[:, 0], others[:, 1]))
        all_players = list(set(all_starters + others))
        if year not in df['year'].values: # new season
            vals = []
            for pos in SEASON_POSITIONS:
                pos_players = [s[0] for s in all_players if s[1] == pos]
                vals.append('|'.join(pos_players))
            new_df.loc[len(new_df.index)] = [year, abbr] + vals
        else: # same season - append new players
            data: pd.DataFrame = df.loc[(df['year']==year)&(df['abbr']==abbr)]
            found_players = '|'.join([d for d in data[cols].values[0] if not pd.isna(d)])
            idx = data.index.values[0]
            for pid, pos in all_players:
                if pid not in found_players:
                    target_col = pos.lower() + 's'
                    try:
                        df.at[idx, target_col] = '|'.join([df.at[idx, target_col], pid])
                    except TypeError:
                        df.at[idx, target_col] = pid
    df = pd.concat([df, new_df])
    df.to_csv("%s.csv" % "allSeasonPlayers", index=False)
    return

def updateAdvancedStats(week, year, _dir): # advanced stat frames
    df = pd.read_csv("%s.csv" % "advancedStats")
    pdf = pd.read_csv("%s.csv" % "allPids")
    a_pids = list(set(df['player_id'].values))
    pids = '|'.join(pdf['pids'].values)
    pids = list(set([p.split(":")[0] for p in pids.split("|")]))
    missing_pids = list(set(pids).difference(set(a_pids)))
    file = open("missingAdvancedStats.txt", "r+")
    none_pids = file.read().split("\n")
    missing_pids = list(set(missing_pids).difference(set(none_pids)))
    curr_pids = df.loc[df['season']==str(year), 'player_id'].values
    missing_pids += list(curr_pids)
    if len(missing_pids) == 0:
        print("advancedStats already up-to-date.")
        return
    df_list = []
    for index, pid in enumerate(missing_pids):
        print(index, len(missing_pids))
        p = Player(pid)
        temp_df = p.dataframe
        if temp_df is None:
            print(f"{pid}: Empty frame")
            file.write('\n' + pid)
        df_list.append(temp_df)
        time.sleep(2)
    file.close()
    new_df = pd.concat(df_list)
    new_df = new_df[
        ['player_id', 'season', 'team_abbreviation']+[col for col in new_df.columns if col not in ['player_id', 'season', 'team_abbreviation']]
    ]
    new_df.drop(columns=['birth_date', 'height', 'weight'], inplace=True)
    df = df.loc[~df['player_id'].isin(new_df['player_id'].values)]
    df = pd.concat([df, new_df])
    df.to_csv("%s.csv" % "advancedStats", index=False)
    return

######################################

def getAllData(week, year):
    
    # make week folder
    _dir = str(week) + "-" + str(year)
    if _dir not in os.listdir("newWeekData/"):
        os.mkdir("newWeekData/" + _dir)
    _dir = "newWeekData/" + _dir
    _dir += "/"
    
    # build raw data
    print()
    print("Building raw data...")
    buildRawData_fixed(week, year, _dir)
    
    # build player data
    print()
    print("Building player data...")
    buildPlayerData(week, year, _dir)
    
    # convert data
    print()
    print("Converting raw data...")
    convertRawData(week, year, _dir)
    
    # scrape starters
    print()
    print("Building starters...")
    buildScrapeStarters(week, year, _dir)
    
    print()
    print('Concatenating new starters with allStarters...')
    concatStarters(week, year, _dir)
    
    # scrape player positions
    print()
    print("Scraping/building player positions...")
    buildPlayerPositions(week, year, _dir)

    # split positions
    print()
    print("Splitting postitions...")
    splitPositions(week, year, _dir)
    
    # clean data
    print()
    print("Cleaning position data...")
    cleanPositionData(week, year, _dir)
    
    # add extra data
    print()
    print("Adding extra stats...")
    addExtraData(week, year, _dir)
    
    # add volume percentages
    print()
    print('Adding volume percentages...')
    addPercentages(week, year, _dir)
    
    # merge positions - LB + DL
    print()
    print("Merging LB + DL...")
    mergePositions(week, year, _dir, 'LB', 'DL')
    
    # concat files
    print()
    print("Concatenating files...")
    concatFiles(week, year, _dir)

    # copy previous week starters
    print()
    print("Copying previous week starters...")
    # !!! USES CURRENT WEEK (week + 1) FOR SOURCE AND NEW WY !!!
    duplicateStarters(week, year, _dir)
    
    # build and concat new fantasy and skill data
    print()
    print("Building/concatenating new fantasy and skill data...")
    concatFantasyAndSkillData(week, year, _dir)
    
    # concat allPids using new week data
    print()
    print('Adding to allPids...')
    allPids(week, year, _dir)
    
    # concat allSeason players using new week data
    print()
    print('Adding to allSeasonPlayers...')
    allSeasonPlayers(week, year, _dir)
    
    # get/add new summaries
    print()
    print('Adding new summaries...')
    getSummaries(week, year, _dir)
    
    # update starts info
    print()
    print('Concatenating new startsInfo...')
    updateStartsInfo("../starters/")
    
    # update advanced stats
    print()
    print('Updating advanced stats...')
    updateAdvancedStats(week, year, _dir)
    av = ApproximateValues("./")
    av.predict()
    
    # update snapCounts
    print()
    sc_dir = "../snapCounts/"
    scm = SCMain(sc_dir)
    scm.update()
    
    # update OLStatsData
    osd = OlStatsData("./")
    osd.update()
    
    return
        
########################################

getAllData(
    week=21, # -> past week
    year=2023
)