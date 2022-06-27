import pandas as pd
import numpy as np
import os

SOURCE_PATH = "../joining/"
DATA_PATH ="../../../../../rawData/positionData/"
RAW_PATH = DATA_PATH.replace("positionData/", "")

def buildAllowedPassingTouchdownsN(n):

    source = pd.read_csv("%s.csv" % (SOURCE_PATH + "source"))
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "QBData"))
    data = pd.read_csv("%s.csv" % (RAW_PATH + "convertedData_78-21W20"))

    cols = [("allowedPassingTouchdowns" + str(n) + "_" + str(i)) for i in range(n)]
    cols = list(source.columns) + cols
    
    df = pd.DataFrame(columns=cols)

    for index, row in source.iterrows():
        pid = row['p_id']
        abbr = row['key'].split("-")[0]
        key = row['key'].split("-")[1]
        sourceCols = [row['key'], row['opp_abbr'], row['wy'], pid]
        opp_abbr = row['opp_abbr']
        start = data.loc[data['key']==key].index.values[0]
        opps = data.loc[(data.index<start)&((data['home_abbr']==opp_abbr)|(data['away_abbr']==opp_abbr))]   
        opps = opps.tail(n)
        home_stats = opps.loc[opps['home_abbr']==opp_abbr, 'away_pass_touchdowns']
        away_stats = opps.loc[opps['away_abbr']==opp_abbr, 'home_pass_touchdowns']
        stats = pd.concat([home_stats, away_stats]).sort_index(ascending=False).values
        if opps.empty:
            emptyArr = np.empty(n)
            emptyArr[:] = np.NaN
            stats = emptyArr
        elif len(opps.index) < n:
            dif = n - len(opps.index)
            emptyArr = np.empty(dif)
            emptyArr[:] = np.NaN
            stats = np.concatenate([stats, emptyArr])
        df.loc[len(df.index)] = sourceCols + list(stats)

    df.fillna(df.mean(), inplace=True)
    df = df.round(0)

    df.to_csv("%s.csv" % ("allowedPassingTouchdowns" + str(n)), index=False)

###############################

buildAllowedPassingTouchdownsN(10)