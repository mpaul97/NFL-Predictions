import pandas as pd
import numpy as np
import os
import random

from sklearn.preprocessing import LabelEncoder

def getRandSimpleOu():
    val = 42
    iod = random.randrange(0, 2)
    steps = random.randrange(0, 30)
    if iod == 0:
        for i in range(steps):
            val -= 0.5
    elif iod == 1:
        for i in range(steps):
            val += 0.5
    oou = random.randrange(0, 2)
    strs = ['o', 'u', 'p']
    randStr = np.random.choice(strs, 3, p=[0.45, 0.45, 0.1])
    return (str(val) + "|" + randStr[0])

def getInfo(row, abbr):
    info = []
    info.append(row['time'])
    info.append(row['surface'])
    info.append(row['roof'])
    info.append(row['stadium'])
    info.append(row['weather'])
    info.append(row['lineHit'])
    # simple vegas line
    vl = row['vegas_line']
    winningName = row['winning_name']
    losingName = row['losing_name']
    winningAbbr = row['winning_abbr']
    losingAbbr = row['losing_abbr']
    if vl != 'Pick':
        if winningName in vl:
            vl = float(vl.replace(winningName, ""))
            targetAbbr = winningAbbr
        elif losingName in vl:
            vl = float(vl.replace(losingName, ""))
            targetAbbr = losingAbbr
        if targetAbbr != abbr:
            vl = np.abs(vl)
    else:
        vl = 0.0
    info.append(vl)
    # end simple vegas line
    info.append(row['month'])
    info.append(row['ouHit'])
    # simple over under
    ou = row['over_under']
    if type(ou) == str:
        temp = ou.split(" ")
        info.append(temp[0] + "|" + temp[1].replace("(","").replace(")","").replace(" ","")[0])
    else:
        info.append(getRandSimpleOu())
    return info

#######################################

DATA_PATH = "../../../../rawData/"

source = pd.read_csv("source.csv")
cd = pd.read_csv("%s.csv" % (DATA_PATH + "convertedData_78-21W20"))

# encode string columns
encode_cols = ['roof', 'stadium']

for col in encode_cols:
    lb = LabelEncoder()
    temp = lb.fit_transform(cd[col])
    cd[col] = temp

cols = ['time', 'surface', 'roof', 'stadium',
        'weather', 'lineHit', 'simpleVl', 'month',
        'ouHit', 'simpleOu']
    
cols = list(source.columns) + cols

df = pd.DataFrame(columns=cols)

for index, row in cd.iterrows():
    key = row['key']
    homeAbbr = row['home_abbr']
    awayAbbr = row['away_abbr']
    # home
    info = getInfo(row, homeAbbr)
    df.loc[-1] = [(homeAbbr + "-" + key), awayAbbr, row['wy']] + info
    df.index = df.index + 1
    df = df.sort_index(ascending=False)
    # away
    info = getInfo(row, awayAbbr)
    df.loc[-1] = [(awayAbbr + "-" + key), homeAbbr, row['wy']] + info
    df.index = df.index + 1
    df = df.sort_index(ascending=False)
    
df = df.reset_index(drop=True)
    
df.to_csv("sourceInfo.csv", index=False)