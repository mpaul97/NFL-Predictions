import pandas as pd
import numpy as np
import os
import time
import regex as re

import urllib.request

import sys
sys.path.append('../')
from myRegex.namesRegex import getNames

def getContent(abbr, year):
    
    url = "https://www.pro-football-reference.com/teams/" + abbr.lower() + "/" + year + ".htm"
    
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()

    mystr = mybytes.decode("utf8", errors='ignore')
    fp.close()
    
    start = [m.start() for m in re.finditer(r"<strong>Coach:</strong>", mystr)][0]
    mystr = mystr[start:start+200]
    
    mystr = re.sub(r"<.*?>", "", mystr)
    
    names = getNames(mystr, False)
    names = [n for n in names if 'Points' not in n]
    
    return names

def getCoaches():
    df = pd.read_csv("%s.csv" % "../data/gameData")
    years = list(set([int(wy.split(" | ")[1]) for wy in df['wy'].values]))
    years.sort()
    new_df = pd.DataFrame(columns=['year', 'abbr', 'coach'])
    for year in years:
        abbrs = list(set(df.loc[df['wy'].str.contains(str(year)), 'home_abbr'].values))
        for abbr in abbrs:
            print(year, abbr)
            coaches = getContent(abbr, str(year))
            new_df.loc[len(new_df.index)] = [year, abbr, '|'.join(coaches)]
            time.sleep(2)
    new_df.to_csv("%s.csv" % "coaches", index=False)
    return

def buildIsNewCoach():
    df = pd.read_csv("%s.csv" % "coaches")
    new_coaches = []
    for index, row in df.iterrows():
        year = row['year']
        abbr = row['abbr']
        coach = row['coach']
        if year > 1994:
            try:
                last_coach = df.loc[(df['year']==(year-1))&(df['abbr']==abbr), 'coach'].values[0]
                new_coaches.append(1 if coach != last_coach else 0)
            except IndexError:
                print(abbr, 'new team')
                new_coaches.append(0)
        else:
            new_coaches.append(0)
    df['isNewCoach'] = new_coaches
    df.to_csv("%s.csv" % "coachInfo", index=False)
    return

##########################

# getCoaches(df) => added 2023 coaches !! dont run

buildIsNewCoach()