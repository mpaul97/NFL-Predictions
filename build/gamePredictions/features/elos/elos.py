import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = "../../../../rawData/"
TARGET_PATH = "../../targets/"

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

def getLastWyEnd(wy):
    week = int(wy.split(" | ")[0])
    year = int(wy.split(" | ")[1])
    if week == 1:
        return str(year)
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

def findEndEloYearly(teamElo):
	return 1500

##########################################

def buildElos(_type):

    df = pd.read_csv("%s.csv" % (TARGET_PATH + "target"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    elos = pd.read_csv("%s.csv" % (DATA_PATH + "initialElos"))

    names = list(elos['name'])

    eloList = []

    for index, row in df.iterrows():
        team = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        temp = cd.loc[cd['key']==key] # get both abbrs -> find opponent abbr
        abbrs = [temp['home_abbr'].values[0], temp['away_abbr'].values[0]]
        abbrs.remove(team)
        opp = abbrs[0]
        wy = row['wy']
        week = wy.split(" | ")[0]
        year = wy.split(" | ")[1]
        if wy != "1 | 1980": # init
            lastWy = getLastWyEnd(wy)
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
                        if _type == "normal":
                            newEloList.append(Elo(row['name'], findEndElo(row[lastRowWy])))
                        elif _type == "yearly":
                            newEloList.append(Elo(row['name'], findEndEloYearly(row[lastRowWy])))
                    elos[year] = [x.elo for x in newEloList]
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

    if _type == 'normal':
        elos.to_csv("elos.csv", index=False)
    elif _type =="yearly":
        elos.to_csv("elosYearly.csv", index=False)

def visualizeElos(abbrs, _type):

    if _type == "normal":
        fn = "elos"
    elif _type == "yearly":
        fn = "elosYearly"

    elos = pd.read_csv("%s.csv" % fn)

    for abbr in abbrs:
        temp = elos.loc[elos['name']==abbr]
        temp = temp.drop(columns=['name'])
        x = list(temp.columns)
        temp = temp.to_numpy()
        y = temp[0]
        plt.plot(x, y)
        
    plt.show()

####################################

def smooth(df, alpha):
    names = df['name']
    df = df.ewm(alpha=alpha).mean()
    df.insert(0, 'name', names)
    return df

def smoothElos(_type, alpha, remove):
            
    if _type == "normal":
        fn = "elos"
    elif _type == "yearly":
        fn = "elosYearly"
        
    if remove:
        for n in os.listdir("."): # remove smooth, replace with current alpha
            if (fn + "Smooth") in n:
                os.remove(n)
    
    df = pd.read_csv("%s.csv" % fn)
    
    df = smooth(df, alpha=alpha)
    
    df.to_csv("%s.csv" % (fn + "Smooth-" + str(alpha)), index=False)

def visualizeElosSmooth(abbr, _type, alpha):
    
    if _type == "normal":
        fn = "elos"
    elif _type == "yearly":
        fn = "elosYearly"
        
    data = pd.read_csv("%s.csv" % (fn + "Smooth-" + str(alpha)))

    elos = pd.read_csv("%s.csv" % fn)
    
    df_list = [elos, data]

    for df in df_list:
        temp = df.loc[df['name']==abbr]
        temp = temp.drop(columns=['name'])
        x = list(temp.columns)
        temp = temp.to_numpy()
        y = temp[0]
        plt.plot(x, y)
        
    plt.show()
    
####################################

# buildElos(_type="normal")

# visualizeElos(abbrs=['GNB', 'KAN'], _type="normal")

ALPHA = 0.65
_type = "normal"

smoothElos(_type, ALPHA, remove=False)

visualizeElosSmooth('GNB', _type, alpha=ALPHA)