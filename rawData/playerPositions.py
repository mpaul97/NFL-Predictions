import pandas as pd
import numpy as np

def cleanPositions():
    
    df = pd.read_csv("%s.csv" % "playerData_78-21W20")
    pos = pd.read_csv("%s.csv" % "truePositions_78-21W20")
    
    temp = list(set(list(pos['position'])))
    
    temp.sort()
    
    sizes = []
    
    for t in temp:
        temp1 = pos.loc[pos['position']==t]
        sizes.append((t, len(temp1.index)))
        
    sizes.sort(key=lambda x: x[1], reverse=True)
    
    new_df = pd.DataFrame()
    new_df['position'] = [s[0] for s in sizes]
    new_df['size'] = [s[1] for s in sizes]
    
    new_df.to_csv("%s.csv" % "positionSizes", index=False)
    # simplePositions -> created manually
    
def splitPositions():
    
    df = pd.read_csv("%s.csv" % "playerData_78-21W20")
    tp = pd.read_csv("%s.csv" % "truePositions_78-21W20")
    sp = pd.read_csv("%s.csv" % "positionSizes")
    
    positions = list(set(list(sp['simplePosition'])))
    
    simpleList = []
    
    for index, row in df.iterrows():
        pid = row['p_id']
        print(row['game_key'], row['wy'])
        position = tp.loc[tp['p_id']==pid, 'position'].values[0]
        simplePosition = sp.loc[sp['position']==position, 'simplePosition'].values[0]
        simpleList.append(simplePosition)
        
    df['position'] = simpleList
    
    for pos in positions:
        temp = df.loc[df['position']==pos]
        temp.to_csv("%s.csv" % ("positionData/" + pos + "Data"), index=False)
    
#############################

splitPositions()