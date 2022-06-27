from fastjsonschema import JsonSchemaDefinitionException
import pandas as pd
import numpy as np

ELO_PATH = "../elos/"
PREDICT_PATH = "../../predict/"

def getLastWyEnd(wy):
    week = int(wy.split(" | ")[0])
    year = int(wy.split(" | ")[1])
    if week == 1:
        return str(year)
    else:
        week -= 1
    if wy == "11 | 1982":
        week = 2
        year = 1982
    if wy == "4 | 1987":
        week = 2
        year = 1987
    return str(week) + " | " + str(year)

def getEloLists(fn, source):
    df = pd.read_csv("%s.csv" % (fn))
    elos, opp_elos = [], []
    for index, row in source.iterrows():
        abbr = row['key'].split("-")[0]
        opp_abbr = row['opp_abbr']
        wy = row['wy']
        lastWy = getLastWyEnd(wy)
        if '1980' in wy:
            temp = df.loc[df['name']==abbr, wy].values[0]
            temp1 = df.loc[df['name']==opp_abbr, wy].values[0]
        else:
            temp = df.loc[df['name']==abbr, lastWy].values[0]
            temp1 = df.loc[df['name']==opp_abbr, lastWy].values[0]
        elos.append(temp)
        opp_elos.append(temp1)
    return elos, opp_elos

def joinNew(featureNames, week, year): # create new predict
    
    source = pd.read_csv("%s.csv" % ("source_w" + str(week)))
    
    for name in featureNames:
        if 'elo' in name:
            elos, opp_elos = getEloLists(ELO_PATH + name, source)
            name = name.replace("s", "")
            source[name] = elos
            source['opp_' + name] = opp_elos
    
    source.to_csv("%s.csv" % (PREDICT_PATH + (str(week) + "-" + str(year)) + "/test" + str(len(featureNames))), index=False)
    
def joinExisting(prevNum, featureName, week, year): # join new feature to existing train
    
    df = pd.read_csv("%s.csv" % (PREDICT_PATH + (str(week) + "-" + str(year)) + "/test" + str(prevNum)))
    
    # update train info
    info = pd.read_csv("%s.csv" % "testInfo")
    temp = [(prevNum + 1), featureName]
    info.loc[len(info.index)] = temp
    info.to_csv("%s.csv" % "testInfo", index=False)
    # end update train info
    
    if 'elo' in featureName:
        elos, opp_elos = getEloLists(ELO_PATH + featureName, df)
        featureName = featureName.replace("s", "")
        df[featureName] = elos
        df['opp_' + featureName] = opp_elos
    elif 'last' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../lastN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy'])
    elif 'time' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../timeN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy'])
    elif 'opp' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../oppN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy'])
    elif 'points' in featureName and 'pointsAllowed' not in featureName:
        df1 = pd.read_csv("%s.csv" % ("../pointsN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy'])
    elif 'pointsAllowed' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../pointsAllowedN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy'])
    else:
        df1 = pd.read_csv("%s.csv" % ("../" + featureName + "/" + featureName))
        df = df.merge(df1)
        
    df.to_csv("%s.csv" % (PREDICT_PATH + (str(week) + "-" + str(year)) + "/test" + str(prevNum + 1)), index=False)
    
###############################

# joinNew(["elos"], week=1, year=2022)

joinExisting(prevNum=1, 
             featureName="elosYearly",
             week=1,
             year=2022
)