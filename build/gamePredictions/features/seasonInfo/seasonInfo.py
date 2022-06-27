import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
TARGET_PATH = "../../targets/"
DATA_PATH = "../../../../rawData/"

def buildInfo():
    
    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    target = pd.read_csv("%s.csv" % (TARGET_PATH + "target"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))
    
    weeks, season, seasonWins, seasonLoses, seasonWinPrec, isHome = [], [], [], [], [], [];
    
    for index, row in target.iterrows():
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        weeks.append(week)
        season.append(year)
        abbr = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        if week == 1:
            seasonWins.append(0)
            seasonLoses.append(0)
            seasonWinPrec.append(0)
        else:
            temp = target.loc[(target.index<index)&(target['key'].str.contains(abbr + "-")), 'won'].values
            seasonWins.append(sum(temp))
            seasonLoses.append(list(temp).count(0))
            seasonWinPrec.append(sum(temp)/len(temp))
        temp1 = cd.loc[cd['key']==key]
        if abbr == temp1['home_abbr'].values[0]:
            isHome.append(1)
        else:
            isHome.append(0)

    source['week'] = weeks
    source['season'] = season
    source['seasonWins'] = seasonWins
    source['seasonLoses'] = seasonLoses
    source['seasonWinPercentage'] = seasonWinPrec
    source['isHome'] = isHome
    
    source.to_csv("%s.csv" % "seasonInfo", index=False)
    
#################################

buildInfo()