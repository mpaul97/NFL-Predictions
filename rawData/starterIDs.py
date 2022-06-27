from sportsipy.nfl.boxscore import Boxscores
from sportsipy.nfl.boxscore import Boxscore
from sportsipy.nfl.teams import Teams

import numpy as np
import csv
import pandas as pd
import urllib

def listToString(temp):
	count = 0
	k = ''
	for l in temp:
		if count < len(temp)-1:
			k += str(l) + " | "
		else:
			k += str(l)
		count += 1
	return k

wy = []

box_list = []

all_players = []

count = 0

# week = 1
# year = 2002

# while year < 2022:
# 	try:
# 		box_list.append(Boxscores(week, year))
# 	except urllib.error.HTTPError as exception:
# 		print("Null Week:", week, year)
# 	week += 1
# 	if week == 21:
# 		week = 1
# 		year += 1

box_list.append(Boxscores(17, 2021))
box_list.append(Boxscores(18, 2021))
box_list.append(Boxscores(19, 2021))
box_list.append(Boxscores(20, 2021))

for boxscores in box_list:
    for key in boxscores.games.keys():
        for item in boxscores.games[key]:

            l1 = list(item.values())#values from boxscores to get game IDs
            print(l1[0])
            boxscore = Boxscore(l1[0])

            h_players = boxscore.home_players
            a_players = boxscore.away_players

            h_abbr = boxscore.home_abbreviation.upper()
            a_abbr = boxscore.away_abbreviation.upper()

            h_list = []
            a_list = []

            for p in h_players:
                h_list.append(p.player_id)

            for p in a_players:
                a_list.append(p.player_id)

            hp_string = listToString(h_list)
            ap_string = listToString(a_list)

            home_df = pd.DataFrame(
                {'abbr' : h_abbr,
                    'wy': wy[count],
                    'players': hp_string,
                    'isHome': True
                }, index=[0])

            away_df = pd.DataFrame(
                {'abbr' : a_abbr,
                    'wy': wy[count],
                    'players': ap_string,
                    'isHome': False
                }, index=[0])

            all_players.append(home_df)
            all_players.append(away_df)

    count += 1

df_all = pd.concat(all_players)

df_all = df_all.reset_index()

df_all = df_all.drop(columns='index')

df_all.fillna(0, inplace=True)

df_all.to_csv('%s.csv' % 'starterIDs_21W16-20')