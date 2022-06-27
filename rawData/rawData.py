from sportsipy.nfl.boxscore import Boxscores
from sportsipy.nfl.boxscore import Boxscore
from sportsipy.nfl.teams import Teams

import numpy as np
import csv
import pandas as pd
import urllib

wy = []

box_list = []
df_list = []

# week = 1
# year = 1978

# while year < 2002:
# 	try:
# 		box_list.append(Boxscores(week, year))
# 	except urllib.error.HTTPError as exception:
# 		print("Null Week:", week, year)
# 	week += 1
# 	if week == 21:
# 		week = 1
# 		year += 1

box_list.append(Boxscores(21, 2021))
box_list.append(Boxscores(22, 2021))

for boxscores in box_list:
	for key in boxscores.games.keys():
		for item in boxscores.games[key]:
			l1 = list(item.values())
			boxscore = Boxscore(l1[0])
			wy.append(key.replace("-", " | "))
			temp_df = boxscore.dataframe
			temp_df['winning_abbr'] = boxscore.winning_abbr
			temp_df['home_abbr'] = boxscore.home_abbreviation.upper()
			temp_df['away_abbr'] = boxscore.away_abbreviation.upper()
			df_list.append(temp_df)

#raw stats
df = pd.concat(df_list)
df['wy'] = wy
df.insert(0, 'key', df.index)
df.to_csv("%s.csv" % "rawData_21_W21-22", index=False)