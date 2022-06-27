import pandas as pd
import numpy as np

SOURCE_PATH = "../joining/"
DATA_PATH = "../../../../rawData/"

def getWeekAfterBye(row, wy, source):
    abbr = row['key'].split("-")[0]
    i = source.loc[source['wy']==wy].index[0]
    prevWy = source.loc[source.index==(i-1), 'wy'].values[0]
    prevAbbrs = list(source.loc[source['wy']==prevWy, 'opp_abbr'].values)
    if abbr in prevAbbrs:
        return 0
    else:
        return 1

def buildPlayoffsInfo():
    
    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    df = pd.read_csv("%s.csv" % (DATA_PATH + "seasonLength-1"))
    
    isPlayoffs, isWeekBeforePlayoffs, isWeekAfterBye = [], [], []
    
    for index, row in source.iterrows():
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        sWeeks = df.loc[df['year']==year, 'weeks'].values[0]
        # is playoffs
        if week > sWeeks:
            isPlayoffs.append(1)
        else:
            isPlayoffs.append(0)
        # is week before playoffs
        if week == sWeeks:
            isWeekBeforePlayoffs.append(1)
        else:
            isWeekBeforePlayoffs.append(0)
        # is week after bye
        if week != 1:
            isWeekAfterBye.append(getWeekAfterBye(row, wy, source))
        else:
            isWeekAfterBye.append(0)
            
    source['isPlayoffs'] = isPlayoffs
    source['isWeekBeforePlayoffs'] = isWeekBeforePlayoffs
    source['isWeekAfterBye'] = isWeekAfterBye
    
    source.to_csv("%s.csv" % "playoffsByeInfo", index=False)
    
###################################

buildPlayoffsInfo()