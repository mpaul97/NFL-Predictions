import sys
sys.path.append("../")

import pandas as pd
import os
from sportsipy.nfl.roster import Player

from paths import POSITION_PATH

def most_common(List):
    return max(set(List), key = List.count)

def addInfo():
    
    df0 = pd.read_csv("%s.csv" % "playerNames")
    df1 = pd.read_csv("%s.csv" % "playerNamesOff")
    df = pd.concat([df0, df1])
    df.drop_duplicates(inplace=True)
    
    cd_list = []
    for fn in os.listdir(POSITION_PATH):
        if 'csv' in fn:
            cd_list.append(pd.read_csv(POSITION_PATH + fn))
    cd = pd.concat(cd_list)
    
    abbrs, positions = [], []
    
    for index, row in df.iterrows():
        print(str(round(index/len(df.index), 2)*100) + "%")
        pid = row['p_id']
        stats = cd.loc[cd['p_id']==pid]
        if not stats.empty:
            position = most_common(list(stats['position'].values))
            abbr = most_common(list(stats['abbr'].values))
        else:
            print(pid)
            position = 'UNK'
            abbr = 'UNK'
        abbrs.append(abbr)
        positions.append(position)
        
    df['position'] = positions
    df['abbr'] = abbrs
    
    df.to_csv("%s.csv" % "playerNames", index=False)
    
    return

def creatingMissing():
    
    df = pd.read_csv("%s.csv" % "playerNames")
    
    mdf = df.loc[df['position']=='UNK']
    
    mdf.to_csv("%s.csv" % "missingPlayerNames", index=False)
    
    return

def combineMissing():
    
    df = pd.read_csv("%s.csv" % "playerNames")
    
    mdf = pd.read_csv("%s.csv" % "missingPlayerNames")
    
    df = df.loc[~df['p_id'].isin(mdf['p_id'].values)]
    
    df = pd.concat([df, mdf])
    
    df.to_csv("%s.csv" % "playerNames", index=False)
    
    return

def addYears():
    
    df = pd.read_csv("%s.csv" % "playerNames")
    cd_list = []
    for fn in os.listdir(POSITION_PATH):
        if 'csv' in fn:
            cd_list.append(pd.read_csv(POSITION_PATH + fn))
    cd = pd.concat(cd_list)
    
    df = df.head(10)
    
    allAbbrYears = []
    
    for index, row in df.iterrows():
        pid = row['p_id']
        info = cd.loc[cd['p_id']==pid, ['abbr', 'wy']].values
        abbrYears = []
        for i in range(info.shape[0]):
            vals = info[i]
            abbrYears.append(vals[0] + ":" + vals[1])
        allAbbrYears.append(','.join(abbrYears))
        
    df['abbrByYear'] = allAbbrYears
    
    df.to_csv("%s.csv" % "temp", index=False)
    
    return

##################

# addInfo()

# creatingMissing()

# combineMissing()

# addYears()

p = Player('RodgAa00')

print(p._team_abbreviation)