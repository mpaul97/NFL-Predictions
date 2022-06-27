import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../build/gamePredictions/features/joining/"
POSITION_PATH = "positionData/"

# qb, qb, rb, rb, rb, wr, wr, wr, wr, wr, te, te

def buildStarters():
    
    cd = pd.read_csv("%s.csv" % "convertedData_78-21W20")
    pdf = pd.read_csv("%s.csv" % (POSITION_PATH + "QBData"))
    
    cd = cd.head(10)
    
    for index, row in cd.iterrows():
        gameKey = row['key']
        home_abbr = row['home_abbr']
        temp = pdf.loc[(pdf['game_key']==gameKey)&(pdf['abbr']==home_abbr)]
        print(temp)
    
######################

buildStarters()