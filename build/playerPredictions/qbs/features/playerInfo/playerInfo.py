import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
DATA_PATH ="../../../../../rawData/positionData/"

def buildPlayerInfo():

    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "QBData"))

    isNewStarter, isNewTeam, outLastSeason, notFullGame = [], [], [], []

    for index, row in source.iterrows():
        pid = row['p_id']
        abbr = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        wy = row['wy']
        game = cd.loc[(cd['p_id']==pid)&(cd['game_key']==key)]
        if not game.empty:
            start = game.index.values[0]
            # isNewStarter
            abbr = game['abbr'].values[0]
            prevGames = cd.loc[(cd.index<start)&(cd['abbr']==abbr)]
            prevGames = prevGames.tail(3)
            prevGames = prevGames.sort_values(by=['attempted_passes'])
            if not prevGames.empty:
                prevPid = prevGames.tail(1)['p_id'].values[0]
                if prevPid != pid:
                    isNewStarter.append(1)
                else:
                    isNewStarter.append(0)
            else:
                isNewStarter.append(1)
            # isNewTeam
            prevAbbr = cd.loc[(cd.index<start)&(cd['p_id']==pid)]
            if prevAbbr.empty or prevAbbr.tail(1)['abbr'].values[0] != abbr:
                isNewTeam.append(1)
            else:
                isNewTeam.append(0)
            # outLastSeason
            prevYear = int(wy.split(" | ")[1])-1
            prevWys = cd.loc[(cd.index<start)&(cd['p_id']==pid)]
            if not prevWys.empty:
                prevWys = prevWys['wy'].values
                prevWys = list(set([int(temp.split(" | ")[1]) for temp in prevWys]))
                if prevYear not in prevWys:
                    outLastSeason.append(1)
                else:
                    outLastSeason.append(0)
            else:
                outLastSeason.append(1)
            # notFullGame
            tempGame = cd.loc[(cd['game_key']==key)&(cd['abbr']==abbr)]
            allPids = tempGame.loc[tempGame['attempted_passes']>4, 'p_id'].values
            if len(allPids) > 1:
                notFullGame.append(1)
            else:
                notFullGame.append(0)
        else:
            isNewStarter.append(1)
            isNewTeam.append(1)
            outLastSeason.append(1)
            notFullGame.append(1)
        
    source['isNewStarter'] = isNewStarter
    source['isNewTeam'] = isNewTeam
    source['outLastSeason'] = outLastSeason
    source['notFullGame'] = notFullGame

    source.to_csv("%s.csv" % "playerInfo", index=False)

########################

buildPlayerInfo()