import pandas as pd
import numpy as np
import os

GAME_PATH = "../../gamePredictions/"
PLAYER_PATH = "../../playerPredictions/"

def alterPlayerCols(pos, temp):
    temp.drop(columns=['p_id'], inplace=True)
    keep_cols = ['key', 'opp_abbr', 'wy']
    cols = list(temp.columns)
    new_cols = [(pos + "_" + col) for col in cols if col not in keep_cols]
    new_cols = keep_cols + new_cols
    temp.columns = new_cols
    return temp

def combine(gameTrainNum, playerTrains):

    # delete old combined
    files = list(os.listdir("."))
    for f in files:
        if 'trainCombined_' in f:
            os.remove(f)

    # load gamePredicitions train csv
    df = pd.read_csv("%s.csv" % (GAME_PATH + "train/train" + str(gameTrainNum)))

    # load each playerPredictions train csv
    playerFn = ""
    for p in playerTrains:
        pos = p[0]
        trainNum = p[1]
        temp = pd.read_csv("%s.csv" % (PLAYER_PATH + pos + "/train/train" + str(trainNum)))
        temp = alterPlayerCols(pos, temp)
        df = df.merge(temp, how='left', on=['key', 'opp_abbr', 'wy'])
        playerFn += pos + str(trainNum) + "_"

    df.to_csv("%s.csv" % ("trainCombined_" + ("g" + str(gameTrainNum)) + "_" + playerFn), index=False)

#########################

playerTrains = [('qbs', 12)]

combine(14, playerTrains)