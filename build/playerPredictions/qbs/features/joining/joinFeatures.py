from fastjsonschema import JsonSchemaDefinitionException
import pandas as pd
import numpy as np

TRAIN_PATH = "../../train/"

def joinNew(featureNames): # create new train
    
    source = pd.read_csv("%s.csv" % "source")
    
    for name in featureNames:
        if 'last' in name:
            df1 = pd.read_csv("%s.csv" % ("../lastN/" + name))
            source = source.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    
    source.to_csv("%s.csv" % (TRAIN_PATH + "train" + str(len(featureNames))), index=False)
    
def joinExisting(prevNum, featureName): # join new feature to existing train
    
    df = pd.read_csv("%s.csv" % (TRAIN_PATH + "train" + str(prevNum)))

    # update train info
    info = pd.read_csv("%s.csv" % "trainInfo")
    temp = [(prevNum + 1), featureName]
    info.loc[len(info.index)] = temp
    info.to_csv("%s.csv" % "trainInfo", index=False)
    # end update train info

    if 'last' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../lastN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    elif 'passingYards' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../passingYardsN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    elif 'passingTouchdowns' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../passingTouchdownsN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    elif 'completedPasses' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../completedPassesN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    elif 'attemptedPasses' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../attemptedPassesN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    elif 'allowedPassingYards' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../allowedPassingYardsN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    elif 'allowedPassingTouchdowns' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../allowedPassingTouchdownsN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    elif 'allowedAttemptedPasses' in featureName:
        df1 = pd.read_csv("%s.csv" % ("../allowedAttemptedPassesN/" + featureName))
        df = df.merge(df1, how='left', on=['key', 'opp_abbr', 'wy', 'p_id'])
    else:
        df1 = pd.read_csv("%s.csv" % ("../" + featureName + "/" + featureName))
        df = df.merge(df1)
        
    df.to_csv("%s.csv" % (TRAIN_PATH + "train" + str(prevNum + 1)), index=False)
    
###############################

# joinNew(["last5"])

joinExisting(prevNum=12, featureName="allowedAttemptedPasses10")