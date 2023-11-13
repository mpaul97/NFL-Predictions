import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import os

from textEncoding.Converter import Converter

NAMES_PATH = "../playerNames/"
ENCODING_PATH = "../textEncoding/"

def linesToNames(_type):
    
    if _type == 's':
        df = pd.read_csv("%s.csv" % (ENCODING_PATH + "scoringSummaries/data/train"))
        
    ndf = pd.read_csv("%s.csv" % (NAMES_PATH + "playerNames"))
    
    lines = list(df['detail'].values)
    
    all_lines = ' . '.join(lines)
    
    c = Converter('', '')
    
    names = c.getNames(all_lines)
    
    new_df = pd.DataFrame()
    
    new_df['name'] = list(set(names))
    
    new_df.to_csv("%s.csv" % ("allNames_" + _type), index=False)
    
    return

def checkNames(_type):
    
    if _type == 's':
        df = pd.read_csv("%s.csv" % ("allNames_" + _type))
        
    ndf = pd.read_csv("%s.csv" % (NAMES_PATH + 'playerNames'))
    
    for index, row in df.iterrows():
        name = row['name']
        if 'safety' not in name.lower():
            try:
                temp = ndf.loc[(ndf['name'].str.contains(name))|(ndf['aka'].str.contains(name))].values[0]
            except IndexError:
                print('MISSING: ' + name)
    
    return

##################

_type='s'

linesToNames(_type)

checkNames(_type)