from sportsipy.nfl.boxscore import Boxscores
from sportsipy.nfl.boxscore import Boxscore
from sportsipy.nfl.teams import Teams

import numpy as np
import csv
import pandas as pd
import urllib

wy = []

all_list = []

week = 1
year = 1978

while year < 2002:
	try:
		all_list.append(Boxscores(week, year))
	except urllib.error.HTTPError as exception:
		print("Null Week:", week, year)
	week += 1
	if week == 21:
		week = 1
		year += 1

# for i in range(12, 16):
# 	all_list.append(Boxscores(i, 2021))
# 	wy.append(str(i) + " | " + "2021")

# all_list.append(Boxscores(20, 2021))
# wy.append("20 | 2021")

# b_2021 = Boxscores(1, 2021, 9) # 6 5 4 3 2 1
# b_2020 = Boxscores(1, 2020, 17) # 17 16 15 14 13 12 11

# box_list = []

# box_list.append(b_2021)
# box_list.append(b_2020)
# box_list.extend(all_list)

all_players = []

count = 0

box_list = all_list

for boxscores in box_list:
	for key in boxscores.games.keys():
		for item in boxscores.games[key]:

			l1 = list(item.values())#values from boxscores to get game IDs
			boxscore = Boxscore(l1[0])

			h_players = boxscore.home_players
			a_players = boxscore.away_players

			h_abbr = boxscore.home_abbreviation.upper()
			a_abbr = boxscore.away_abbreviation.upper()

			for p in h_players:
				df_p = p.dataframe
				df_p['abbr'] = h_abbr
				# df_p['wy'] = wy[count]
				wy.append(key.replace("-", " | "))
				df_p['game_key'] = l1[0]
				df_p['isHome'] = True
				all_players.append(df_p)

			for p in a_players:
				df_p = p.dataframe
				df_p['abbr'] = a_abbr
				# df_p['wy'] = wy[count]
				wy.append(key.replace("-", " | "))
				df_p['game_key'] = l1[0]
				df_p['isHome'] = False
				all_players.append(df_p)
	count += 1

df_all = pd.concat(all_players)

df_all['wy'] = wy

df_all.fillna(0, inplace=True)

df_all = df_all.reset_index()

df_all = df_all.rename(columns={'index': 'p_id'})

# # indexes = list(df_all.index)
# # keys = list(df_all['key'])

# # k1 = pd.DataFrame({'index' : indexes, 'key' : keys})

# # k1 = k1.groupby(['index']).first()

# # first_keys = list(k1['key'])

# # df_sum = df_all.groupby(df_all.index).sum()

# # averages = df_sum.apply(lambda x: x/10)

# # averages['key'] = first_keys

df_all.to_csv('%s.csv' % 'playerData_78-01', index=False)