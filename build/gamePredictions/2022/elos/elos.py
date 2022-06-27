import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"

class Elo:
    def __init__(self, name, elo):
        self.name = name
        self.elo = elo
    def as_dict(self):
	    return {'name': self.name, 'elo': self.elo}

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

def buildElos(_type, week, year):
    
    wy = str(week) + " | " + str(year)
    
    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source_w" + str(week)))
    
    if _type == "normal":
        elos = pd.read_csv("%s.csv" % "elos")
    elif _type == "yearly":
        elos = pd.read_csv("%s.csv" % "elosYearly")
        
    eloList = []
    lastWy = getLastWyEnd(wy) # new column name
        
    for index, row in source.iterrows():
        team_abbr = row['key'].split("-")[0]
        opp_abbr = row['opp_abbr']
        lastRowWy = elos.columns[len(elos.columns)-1] # last column in elos
        teamElo = elos.loc[elos['name']==team_abbr, lastRowWy].values[0]
        oppElo = elos.loc[elos['name']==opp_abbr, lastRowWy].values[0]
        if week == 1:
            if _type == "normal":
                eloList.append(Elo(team_abbr, findEndElo(teamElo)))
            elif _type == "yearly":
               eloList.append(Elo(team_abbr, findEndEloYearly(teamElo)))
        else:
            print() # code for middle weeks
            
    eloList.sort(key = lambda x: x.name)
    elos[lastWy] = [e.elo for e in eloList]
        
    if _type == 'normal':
        elos.to_csv("elos.csv", index=False)
    elif _type =="yearly":
        elos.to_csv("elosYearly.csv", index=False)

# smooth

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

#########################

buildElos(_type="yearly",
          week=1, 
          year=2022
)