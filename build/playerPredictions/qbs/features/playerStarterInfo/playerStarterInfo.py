import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder

SOURCE_PATH = "../joining/"
INFO_PATH = "../../../../../rawData/"
DATA_PATH ="../../../../../rawData/positionData/"

def buildStarterInfo():

    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    info = pd.read_csv("%s.csv" % (INFO_PATH + "qbStarterInfo"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "QBData"))

    # encode awards
    lb = LabelEncoder()
    temp = lb.fit_transform(info['awards'])
    info['awards'] = temp

    cols = ['isHome', 'careerStarts', 'careerGamesPlayedIn', 'isRookie', 'age', 'pastSeasonStarts', 
            'pastSeasonGamesPlayedIn', 'pastSeasonPasserRating', 'pastSeason4thComebacks', 
            'pastSeasonGameWinningDrives', 'pastSeasonAwards']

    cols = list(source.columns) + cols
    
    df = pd.DataFrame(columns=cols)

    for index, row in source.iterrows():
        pid = row['p_id']
        key = row['key'].split("-")[1]
        sourceCols = [row['key'], row['opp_abbr'], row['wy'], pid]
        year = int(row['wy'].split(" | ")[1])
        if pid != 'UNK':
            game = cd.loc[(cd['p_id']==pid)&(cd['game_key']==key)]
            currInfo = info.loc[(info['p_id']==pid)&(info['year']==year)]
            start = currInfo.index.values[0]
            pastInfo = info.loc[(info.index<start)&(info['p_id']==pid)]
            pastSeasonInfo = info.loc[(info['year']==(year-1))&(info['p_id']==pid)]
            # isHome
            isHome = int(game['isHome'].values[0])
            # careerStarts
            careerStarts = sum(pastInfo['gamesStarted'].values)
            # careerGamesPlayedIn/isRookie
            careerGamesPlayedIn = sum(pastInfo['gamesPlayedIn'].values)
            isRookie = 0
            if careerGamesPlayedIn == 0:
                isRookie = 1
            # age
            age = currInfo['age'].values[0]
            # pastSeasonStarts
            if len(pastSeasonInfo.index) > 0:
                pastSeasonStarts = pastSeasonInfo['gamesStarted'].values[0]
            else:
                pastSeasonStarts = 0
            # pastSeasonGamesPlayedIn
            if len(pastSeasonInfo.index) > 0:
                pastSeasonGamesPlayedIn = pastSeasonInfo['gamesPlayedIn'].values[0]
            else:
                pastSeasonGamesPlayedIn = 0
            # pastSeasonPasserRating
            if len(pastSeasonInfo.index) > 0:
                pastSeasonPasserRating = pastSeasonInfo['passerRating'].values[0]
            else:
                pastSeasonPasserRating = 0
            # pastSeason4thComebacks
            if len(pastSeasonInfo.index) > 0:
                pastSeason4thComebacks = pastSeasonInfo['4th_quarter_comebacks'].values[0]
            else:
                pastSeason4thComebacks = 0
            # pastSeasonGameWinningDrives
            if len(pastSeasonInfo.index) > 0:
                pastSeasonGameWinningDrives = pastSeasonInfo['game_winning_drives'].values[0]
            else:
                pastSeasonGameWinningDrives = 0
            # pastSeasonAwards
            if len(pastSeasonInfo.index) > 0:
                pastSeasonAwards = pastSeasonInfo['awards'].values[0]
            else:
                pastSeasonAwards = 0
        else: # UNK -> pid
            print(sourceCols)
            # isHome
            if index % 2 == 0:
                isHome = 1
            else:
                isHome = 0
            # other
            careerStarts = 0
            careerGamesPlayedIn = 0
            isRookie = 0
            age = 25 # 'average' for UNK
            pastSeasonStarts = 0
            pastSeasonGamesPlayedIn = 0
            pastSeasonPasserRating = 0
            pastSeason4thComebacks = 0
            pastSeasonGameWinningDrives = 0
            pastSeasonAwards = 0
        df.loc[len(df.index)] = sourceCols + [isHome, careerStarts, careerGamesPlayedIn,
                                              isRookie, age, pastSeasonStarts, pastSeasonGamesPlayedIn,
                                              pastSeasonPasserRating, pastSeason4thComebacks, pastSeasonGameWinningDrives,
                                              pastSeasonAwards]
        
    df.fillna(0, inplace=True)

    df.to_csv("%s.csv" % "playerStarterInfo", index=False)

#################################

buildStarterInfo()