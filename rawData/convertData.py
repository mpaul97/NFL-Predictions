import pandas as pd

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
	temp = stadium.split(" ")[0]
	sum_id = 0
	for c in temp:
		sum_id += ord(c)
	return sum_id

def getLineHit(row, dfan):
	line = row['vegas_line']
	if "Pick" not in line:
		temp = line.split("-")
		name = temp[0].replace(" ", "")
		if name == "WashingtonRedskins":
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

df = pd.read_csv("%s.csv" % "rawData_78-01")
#convert data
dfan = pd.read_csv("abbrWithNames.csv")

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

df.to_csv("%s.csv" % "convertedData_78-01", index=False)