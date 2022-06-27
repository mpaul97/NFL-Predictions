import pandas as pd
import numpy as np
import os

TRAININFO_PATH = "../../features/joining/"

# take train info
# loop through features
# call each function to build each feature
# use join all to make test.csv quickly/easily

def buildAll():
    
    trainInfo = pd.read_csv("%s.csv" % (TRAININFO_PATH + "trainInfo"))
    
    # placeholder -> only features programmed so far
    trainInfo = trainInfo.head(4)
    
    for index, row in trainInfo.iterrows():
        print(row['featureName'])
    
##################################

buildAll()
