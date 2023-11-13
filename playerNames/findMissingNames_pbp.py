import pandas as pd
import numpy as np
import os
import sys
sys.path.append("../")

from myRegex.namesRegex import getNames

RAW_PBP = pd.read_csv("../playByPlay/data/rawTrain.csv")

def getMissingNames():
    
    df = RAW_PBP.head(10)
    
    for index, row in df.iterrows():
        line = row['detail']
        names = getNames(line, False)
        print(names)
    
    return

########################

getMissingNames()