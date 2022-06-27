import pandas as pd

# name	opp_abbr	wy	
# team_elo	opp_elo	
# won	
# isHome	
# last10	
# last20	
# yearElos_0	yearElos_1	
# last5 Time
# last5 Opponent
# last2

# off_passing	off_rushing	off_total	ppg	def_passing	def_rushing	def_total	ppgAgainst


###########################################
# HELPERS

class Elo:
	def __init__(self, name, elo):
		self.name = name
		self.elo = elo
	def as_dict(self):
		return {'name': self.name, 'elo': self.elo}

def printElo(elos):
	for e in elos:
		print(e.as_dict())

#############

def getLastWy(wy):
	week = int(wy.split(" | ")[0])
	year = int(wy.split(" | ")[1])
	if week == 1:
		week = 20
		year -= 1
	else:
		week -= 1
	if wy == "11 | 1982":
		week = 2
		year = 1982
	if wy == "4 | 1987":
		week = 2
		year = 1987
	return str(week) + " | " + str(year)

def getK(movWinner, eloDifWinner):
	top = (movWinner + 3) ** 0.8
	bottom = 7.5 + (0.006 * (eloDifWinner))
	return 20 * (top/bottom)

def getETeam(oppElo, teamElo):
	dif = oppElo - teamElo
	x = 10 ** (dif/400)
	return 1 / (1 + x)

def getNewElo(k, sTeam, eTeam, teamElo):
	return k * (sTeam - eTeam) + teamElo

def findElo(teamElo, oppElo, sTeam, movWinner, eloDifWinner):
	k = getK(movWinner, eloDifWinner)
	eTeam = getETeam(oppElo, teamElo)
	return k * (sTeam - eTeam) + teamElo

def findEndElo(teamElo):
	if teamElo != 1500:
		return (teamElo * 0.75) + (0.25 * 1505)
	return 1500

def replaceCols(row, isHome, N):
	temp = row.to_frame()
	temp = temp.transpose()
	if isHome:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('home_', str(N) + '_').replace('away_', str(N) + '_opp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	else:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('away_', str(N) + '_').replace('home_', str(N) + '_opp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	return temp

def getHome(abbr, row):
	isHome = False
	if row['winning_abbr'] == abbr:
		if row['winner'] == "Home":
			isHome = True
		else:
			isHome = False
	if row['losing_abbr'] == abbr:
		if row['winner'] == "Home":
			isHome = False
		else:
			isHome = True
	return isHome

def getStats(abbr, name, index, stats, N):
	count = 0
	df_list = []
	for i in range(index - 1, 0, -1):
		if count == N:
			break
		row = stats.iloc[i]
		if row['winning_abbr'] == abbr or row['losing_abbr'] == abbr:
			df_list.append(replaceCols(row, getHome(abbr, row), N))
			count += 1
	df = pd.concat(df_list)
	if count < N:
		count = len(df.index)
	cols = df.columns
	df[cols] = df[cols].apply(pd.to_numeric, errors='ignore', axis=1)
	df = df.sum(numeric_only=True)
	df = df.to_frame()
	df = df.transpose()

	df = df.drop(columns=['attendance'])

	df = df.apply(lambda x: x/count)
	df.insert(0, 'name', name)

	return df

def replaceColsFast(temp, isHome, N):
	if isHome:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('home_', str(N) + '').replace('away_', str(N) + 'opp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	else:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('away_', str(N) + '').replace('home_', str(N) + 'opp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	return temp

def replaceColsTime(temp, isHome):
	if isHome:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('home_', 'Time').replace('away_', 'Timeopp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	else:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('away_', 'Time').replace('home_', 'Timeopp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	return temp

def replaceColsOpp(temp, isHome):
	if isHome:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('home_', 'Opp').replace('away_', 'Oppopp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	else:
		col_names = list(temp.columns)
		for name in col_names:
			new_col_name = name.replace('away_', 'Opp').replace('home_', 'Oppopp_')
			temp = temp.rename(columns={name: new_col_name})
			# temp = temp.rename(index={temp.last_valid_index(): abbr})
	return temp

###########################################

def buildSource():

	df = pd.read_csv("convertedData_78-21W20.csv")

	start_index = df.loc[df['wy'].str.contains("1980")].index.values[0]

	names = []
	oppAbbrs = []
	wys = []

	for index, row in df.iterrows():
		if index >= start_index:
			# HOME
			home_abbr = row['home_abbr']
			name = home_abbr + " - " + row['key']
			names.append(name)
			oppAbbrs.append(row['away_abbr'])
			wys.append(row['wy'])
			# AWAY
			away_abbr = row['away_abbr']
			name = away_abbr + " - " + row['key']
			names.append(name)
			oppAbbrs.append(row['home_abbr'])
			wys.append(row['wy'])

	new_df = pd.DataFrame()

	new_df['name'] = names
	new_df['opp_abbr'] = oppAbbrs
	new_df['wy'] = wys

	new_df.to_csv("purple_Source.csv", index=False)

def buildElos():

	df = pd.read_csv("purple_Source.csv")
	cd = pd.read_csv("convertedData_78-21W20.csv")
	elos = pd.read_csv("initialElos.csv")

	names = list(elos['name'])

	# df = df.loc[df['wy'].str.contains("1980")|df['wy'].str.contains("1981")]

	eloList = []

	for index, row in df.iterrows():
		team = row['name'].split(" - ")[0]
		opp = row['opp_abbr']
		key = row['name'].split(" - ")[1]
		wy = row['wy']
		week = wy.split(" | ")[0]
		year = wy.split(" | ")[1]
		if wy != "1 | 1980": # init
			lastWy = getLastWy(wy)
			lastRowWy = df.iloc[index - 1].wy
			# NEW WEEK
			if week != lastRowWy.split(" | ")[0] and row['wy'] != "2 | 1980":
				if len(eloList) != 32:
					for name in names:
						if name not in [x.name for x in eloList]:
							eloList.append(Elo(name, elos.loc[elos['name']==name, getLastWy(lastRowWy)].values[0]))
				eloList.sort(key = lambda x: x.name)
				elos[lastRowWy] = [x.elo for x in eloList]
				eloList.clear()
				# NEW YEAR
				if year != lastRowWy.split(" | ")[1]:
					newEloList = []
					for index, row in elos.iterrows():
						newEloList.append(Elo(row['name'], findEndElo(row[lastRowWy])))
					elos[year] = (x.elo for x in newEloList)
			# TEAM ELO
			teamElo = elos.loc[elos['name']==team, lastWy]
			if teamElo.empty: # get last last wy if team not present
				teamElo = elos.loc[elos['name']==team, getLastWy(lastWy)].values[0]
			else:
				teamElo = teamElo.values[0]
			# OPP ELO
			oppElo = elos.loc[elos['name']==opp, lastWy]
			if oppElo.empty: # get last last wy if team not present
				oppElo = elos.loc[elos['name']==opp, getLastWy(lastWy)].values[0]
			else:
				oppElo = oppElo.values[0]
			game = cd.loc[cd['key']==key]
			winner = game['winning_abbr'].values[0]
			if team == winner:
				sTeam = 1
				winningElo = teamElo
				losingElo = oppElo
			else:
				sTeam = 0
				winningElo = oppElo
				losingElo = teamElo
			mov = abs(game['home_points'].values[0] - game['away_points'].values[0])
			eloDifWinner = winningElo - losingElo
			# if int(week) != 1:
			newElo = findElo(teamElo, oppElo, sTeam, mov, eloDifWinner)
			# else:
			# 	newElo = findEndElo(teamElo)
			eloList.append(Elo(team, newElo))

	# END FOR

	elos.to_csv("newElos_80-21W20.csv", index=False)

def joinElos():

	source = pd.read_csv("%s.csv" % "purple_Source")
	elos = pd.read_csv("%s.csv" % "cleanElos_80-21W20")

	teamElos = []
	oppElos = []

	for index, row in source.iterrows():
		abbr = row['name'].split(" - ")[0]
		oppAbbr = row['opp_abbr']
		wy = row['wy']
		ii = elos.loc[elos['name']==abbr.lower()].index.values[0]
		teamE = elos.loc[elos.index==ii, wy].values[0]
		iio = elos.loc[elos['name']==oppAbbr.lower()].index.values[0]
		oppE = elos.loc[elos.index==iio, wy].values[0]
		teamElos.append(teamE)
		oppElos.append(oppE)

	source['team_elo'] = teamElos
	source['opp_elo'] = oppElos

	source.to_csv("%s.csv" % "purple0", index=False)

def buildWonAndHome():

	df = pd.read_csv("purple0.csv")
	cd = pd.read_csv("convertedData_78-21W20.csv")

	won = []
	isHome = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		key = row['name'].split(" - ")[1]
		game = cd.loc[cd['key']==key]
		winningAbbr = game['winning_abbr'].values[0]
		homeAbbr = game['home_abbr'].values[0]
		# won
		if abbr == winningAbbr:
			won.append(1)
		else:
			won.append(0)
		# home
		if abbr == homeAbbr:
			isHome.append(1)
		else:
			isHome.append(0)

	df['won'] = won
	df['isHome'] = isHome

	df.to_csv("purple0a.csv", index=False)

def buildLastN(N):

	# df = pd.read_csv("%s.csv" % "purple_Source")
	if N == 10:
		df = pd.read_csv("purple0a.csv")
		newFn = "purple0b"
	elif N == 20:
		df = pd.read_csv("purple0b.csv")
		newFn = "purple1"
	elif N == 2:
		df = pd.read_csv("purple4.csv")
		newFn = "purple5"
	cd = pd.read_csv("%s.csv" % "convertedData_78-21W20")

	names = []

	col_drops = ['attendance', 'surface', 'time',
				 'stadium_id', 'lineHit', 'month',
				 'ouHit', 'duration']

	cd.drop(columns=col_drops, inplace=True)

	statsList = []

	for index, row in df.iterrows():
		names.append(row['name'])
		abbr = row['name'].split(" - ")[0]
		key = row['name'].split(" - ")[1]
		startIndex = cd.loc[cd['key']==key].index[0]
		stats = cd.loc[(cd.index<startIndex)&((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))].tail(N)
		if stats.empty:
			statsH = cd.loc[cd.index<startIndex].tail(20)
			statsH = replaceColsFast(statsH, True, N)
			statsA = cd.loc[cd.index<startIndex].tail(20)
			statsA = replaceColsFast(statsA, False, N)
			stats = pd.concat([statsH, statsA])
		else:
			homeStats = stats.loc[stats['home_abbr']==abbr]
			homeStats = replaceColsFast(homeStats, True, N)
			awayStats = stats.loc[stats['away_abbr']==abbr]
			awayStats = replaceColsFast(awayStats, False, N)
			stats = pd.concat([homeStats, awayStats])
		num = len(stats.index)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/num)
		statsList.append(stats)

	new_df = pd.concat(statsList)

	new_df.insert(0, 'name', names)

	df = df.merge(new_df)

	df.to_csv("%s.csv" % newFn, index=False)

def joinElosYearly():

	source = pd.read_csv("%s.csv" % "purple1")
	elos = pd.read_csv("%s.csv" % "cleanElosYearly_80-21W20")

	teamElos = []
	oppElos = []

	for index, row in source.iterrows():
		abbr = row['name'].split(" - ")[0]
		oppAbbr = row['opp_abbr']
		wy = row['wy']
		ii = elos.loc[elos['name']==abbr.lower()].index.values[0]
		teamE = elos.loc[elos.index==ii, wy].values[0]
		iio = elos.loc[elos['name']==oppAbbr.lower()].index.values[0]
		oppE = elos.loc[elos.index==iio, wy].values[0]
		teamElos.append(teamE)
		oppElos.append(oppE)

	source['team_eloYearly'] = teamElos
	source['opp_eloYearly'] = oppElos

	source.to_csv("%s.csv" % "purple2", index=False)

def buildLastTime5():

	df = pd.read_csv("%s.csv" % "purple2")
	cd = pd.read_csv("%s.csv" % "convertedData_78-21W20")

	names = []

	col_drops = ['attendance', 'surface',
				 'stadium_id', 'lineHit', 'month',
				 'ouHit', 'duration']

	cd.drop(columns=col_drops, inplace=True)

	statsList = []

	for index, row in df.iterrows():
		names.append(row['name'])
		abbr = row['name'].split(" - ")[0]
		key = row['name'].split(" - ")[1]
		startIndex = cd.loc[cd['key']==key].index[0]
		time = cd.iloc[startIndex].time
		stats = cd.loc[(cd.index<startIndex)&(cd['time']==time)&((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))].tail(5)
		if stats.empty: #new team
			statsH = cd.loc[(cd.index<startIndex)&(cd['time']==time)].tail(20)
			statsH = replaceColsTime(statsH, True)
			statsA = cd.loc[(cd.index<startIndex)&(cd['time']==time)].tail(20)
			statsA = replaceColsTime(statsA, False)
			stats = pd.concat([statsH, statsA])
		else:
			homeStats = stats.loc[stats['home_abbr']==abbr]
			homeStats = replaceColsTime(homeStats, True)
			awayStats = stats.loc[stats['away_abbr']==abbr]
			awayStats = replaceColsTime(awayStats, False)
			stats = pd.concat([homeStats, awayStats])
		num = len(stats.index)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/num)
		statsList.append(stats)

	new_df = pd.concat(statsList)

	new_df.drop(columns="time", inplace=True)

	new_df.insert(0, 'name', names)

	df = df.merge(new_df)

	df.to_csv("%s.csv" % "purple3", index=False)

def buildLastOpp5():

	df = pd.read_csv("%s.csv" % "purple3")
	cd = pd.read_csv("%s.csv" % "convertedData_78-21W20")

	names = []

	col_drops = ['attendance', 'surface', 'time',
				 'stadium_id', 'lineHit', 'month',
				 'ouHit', 'duration']

	cd.drop(columns=col_drops, inplace=True)

	statsList = []

	for index, row in df.iterrows():
		names.append(row['name'])
		abbr = row['name'].split(" - ")[0]
		key = row['name'].split(" - ")[1]
		startIndex = cd.loc[cd['key']==key].index[0]
		opp = row['opp_abbr']
		stats = cd.loc[(cd.index<startIndex)&(((cd['home_abbr']==abbr)&(cd['away_abbr']==opp))|((cd['home_abbr']==opp)&(cd['away_abbr']==abbr)))].tail(5)
		if stats.empty: #new team
			statsH = cd.loc[cd.index<startIndex].tail(20)
			statsH = replaceColsOpp(statsH, True)
			statsA = cd.loc[cd.index<startIndex].tail(20)
			statsA = replaceColsOpp(statsA, False)
			stats = pd.concat([statsH, statsA])
		else:
			homeStats = stats.loc[stats['home_abbr']==abbr]
			homeStats = replaceColsOpp(homeStats, True)
			awayStats = stats.loc[stats['away_abbr']==abbr]
			awayStats = replaceColsOpp(awayStats, False)
			stats = pd.concat([homeStats, awayStats])
		num = len(stats.index)
		stats = stats.sum(numeric_only=True).to_frame().transpose()
		stats = stats.apply(lambda x: x/num)
		statsList.append(stats)

	new_df = pd.concat(statsList)

	new_df = new_df.round(1)

	new_df.insert(0, 'name', names)

	df = df.merge(new_df)

	df.to_csv("%s.csv" % "purple4", index=False)

def buildRankings():

	df = pd.read_csv("%s.csv" % "purple5")

	cd = pd.read_csv("%s.csv" % "convertedData_78-21W20")

	target_cols = ['key', 'wy', 'home_pass_yards', 'home_rush_yards', 'home_total_yards',
				   'home_points', 'away_pass_yards', 'away_rush_yards', 'away_total_yards',
				   'away_points', 'home_abbr', 'away_abbr']

	for col in cd.columns:
		if col not in target_cols:
			cd.drop(columns=col, inplace=True)

	names = []
	wys = []
	offP = []
	offR = []
	offT = []
	ppg = []
	defP = []
	defR = []
	defT = []
	ppgA = []

	for index, row in df.iterrows():
		abbr = row['name'].split(" - ")[0]
		key = row['name'].split(" - ")[1]
		wy = row['wy']
		week = int(wy.split(" | ")[0])
		year = int(wy.split(" | ")[1])
		if week > 1:
			start_index = cd.loc[cd['key']==key].index[0]
			temp = cd.loc[(cd.index<start_index)&(cd['wy'].str.contains(str(year)))&((cd['home_abbr']==abbr)|(cd['away_abbr']==abbr))]
			homeT = temp.loc[temp['home_abbr']==abbr]
			awayT = temp.loc[temp['away_abbr']==abbr]
			op = 0
			orr = 0
			ot = 0
			oppg = 0
			dp = 0
			drr = 0
			dt = 0
			dppg = 0
			if not homeT.empty and not awayT.empty:
				#pass offense
				op = homeT['home_pass_yards'].sum() + awayT['away_pass_yards'].sum()
				op = op/(len(homeT.index)+len(awayT.index))
				#rush offense
				orr = homeT['home_rush_yards'].sum() + awayT['away_rush_yards'].sum()
				orr = orr/(len(homeT.index)+len(awayT.index))
				#total offense
				ot = homeT['home_total_yards'].sum() + awayT['away_total_yards'].sum()
				ot = ot/(len(homeT.index)+len(awayT.index))
				#ppg
				oppg = homeT['home_points'].sum() + awayT['away_points'].sum()
				oppg = oppg/(len(homeT.index)+len(awayT.index))
				#pass defense
				dp = homeT['away_pass_yards'].sum() + awayT['home_pass_yards'].sum()
				dp = dp/(len(homeT.index)+len(awayT.index))
				#rush defense
				drr = homeT['away_rush_yards'].sum() + awayT['home_rush_yards'].sum()
				drr = drr/(len(homeT.index)+len(awayT.index))
				#total defense
				dt = homeT['away_total_yards'].sum() + awayT['home_total_yards'].sum()
				dt = dt/(len(homeT.index)+len(awayT.index))
				#ppg against
				dppg = homeT['away_points'].sum() + awayT['home_points'].sum()
				dppg = dppg/(len(homeT.index)+len(awayT.index))
			elif homeT.empty and not awayT.empty:
				#pass offense
				op = awayT['away_pass_yards'].sum()
				op = op/len(awayT.index)
				#rush offense
				orr = awayT['away_rush_yards'].sum()
				orr = orr/len(awayT.index)
				#total offense
				ot = awayT['away_total_yards'].sum()
				ot = ot/len(awayT.index)
				#ppg
				oppg = awayT['away_points'].sum()
				oppg = oppg/len(awayT.index)
				#pass defense
				dp = awayT['home_pass_yards'].sum()
				dp = dp/len(awayT.index)
				#rush defense
				drr = awayT['home_rush_yards'].sum()
				drr = drr/len(awayT.index)
				#total defense
				dt = awayT['home_total_yards'].sum()
				dt = dt/len(awayT.index)
				#ppg against
				dppg = awayT['home_points'].sum()
				dppg = dppg/len(awayT.index)
			elif not homeT.empty and awayT.empty:
				#pass offense
				op = homeT['home_pass_yards'].sum()
				op = op/len(homeT.index)
				#rush offense
				orr = homeT['home_rush_yards'].sum()
				orr = orr/len(homeT.index)
				#total offense
				ot = homeT['home_total_yards'].sum()
				ot = ot/len(homeT.index)
				#ppg
				oppg = homeT['home_points'].sum()
				oppg = oppg/len(homeT.index)
				#pass defense
				dp = homeT['away_pass_yards'].sum()
				dp = dp/len(homeT.index)
				#rush defense
				drr = homeT['away_rush_yards'].sum()
				drr = drr/len(homeT.index)
				#total defense
				dt = homeT['away_total_yards'].sum()
				dt = dt/len(homeT.index)
				#ppg against
				dppg = homeT['away_points'].sum()
				dppg = dppg/len(homeT.index)
			names.append(row['name'])
			wys.append(wy)
			offP.append(op)
			offR.append(orr)
			offT.append(ot)
			ppg.append(oppg)
			defP.append(dp)
			defR.append(drr)
			defT.append(dt)
			ppgA.append(dppg)
		else:
			names.append(row['name'])
			wys.append(wy)
			offP.append(275)
			offR.append(100)
			offT.append(350)
			ppg.append(24)
			defP.append(275)
			defR.append(100)
			defT.append(350)
			ppgA.append(24)

	new_df = pd.DataFrame()

	new_df['name'] = names
	# new_df['wy'] = wys
	new_df['off_passing'] = offP
	new_df['off_rushing'] = offR
	new_df['off_total'] = offT
	new_df['ppg'] = ppg
	new_df['def_passing'] = defP
	new_df['def_rushing'] = defR
	new_df['def_total'] = defT
	new_df['ppgAgainst'] = ppgA

	# new_df.drop(columns=['name'], inplace=True)

	df = df.merge(new_df)

	df.to_csv("%s.csv" % "purple6", index=False)

#############################################
# CALLS

# buildSource()

# joinElos()

# buildWonAndHome()

# buildLastN(10)

# buildLastN(20)

# joinElosYearly()

# buildLastTime5()

# buildLastOpp5()

# buildLastN(2)

# buildRankings()

df = pd.read_csv("purple6.csv")

df = df.round(1)

df.to_csv("purple6.csv", index=False)

