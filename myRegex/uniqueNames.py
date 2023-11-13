import pandas as pd
import numpy as np
import os
import regex as re

NAMES_PATH = "../../playerNames/"

def getUniqueNames():
    
    df = pd.read_csv("%s.csv" % (NAMES_PATH + "playerNames"))
    
    names = df['name'].values
    
    for name in names:
        # if not re.search(r"[A-Z][a-z]+\s[A-Z][a-z]+", name):
        #     if re.search(r"[A-Z][a-z]+[A-Z][a-z]+", name):
        #         print(name)
        if re.search(r"[A-Z][a-z]+[A-Z][a-z]+\s", name):
            print(name)
    
    return

###########################

getUniqueNames()