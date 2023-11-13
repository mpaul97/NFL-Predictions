import pandas as pd
import os
import regex as re

DATA_PATH = "../playByPlay/data/"

def getTeamNames():
    
    df = pd.read_csv("%s.csv" % (DATA_PATH + "rawTrain"))
    df.fillna('', inplace=True)
    df = df.loc[df['detail'].str.contains('Timeout')]
    
    all_names = []
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        line = row['detail']
        try:
            names = re.findall(r"\b[A-Z][a-z]{1,}\b(?:\s+[A-Z][a-z]{1,}\b)*", line)
            name = [n for n in names if n != 'Timeout'][0]
            all_names.append(name)
        except IndexError:
            print(line)
        
    all_names = list(set(all_names))
    all_names.sort()
        
    new_df = pd.DataFrame()
    new_df['name'] = all_names
    
    new_df.to_csv("%s.csv" % "teamNames_pbp", index=False)
    
    return

def testTeamNames():
    
    df = pd.read_csv("%s.csv" % (DATA_PATH + "rawTrain"))
    df.fillna('', inplace=True)
    df = df.loc[df['detail'].str.contains('Timeout')]
    df.reset_index(drop=True, inplace=True)
    
    tdf = pd.read_csv("%s.csv" % "teamNames_pbp")
    
    for index, row in df.iterrows():
        line = row['detail']
        try:
            names = re.findall(r"\b[A-Z][a-z]{1,}\b(?:\s+[A-Z][a-z]{1,}\b)*", line)
            name = [n for n in names if n != 'Timeout'][0]
            abbr = tdf.loc[tdf['names'].str.contains(name), 'abbr'].values
            print(line, name, abbr)
        except IndexError:
            print(line)
        
    return

####################

# getTeamNames()

testTeamNames()