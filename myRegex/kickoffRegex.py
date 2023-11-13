import pandas as pd
import numpy as np
import os
import regex as re
from ordered_set import OrderedSet

def buildTestData():
    
    df = pd.read_csv("%s.csv" % "pbpTestData")
    
    keys = list(OrderedSet(df['key'].values))
    
    new_lines = []
    
    for key in keys:
        lines = df.loc[df['key']==key, 'Detail'].values
        line = lines[0]
        new_lines.append(line)
        
    new_df = pd.DataFrame()
    new_df['line'] = new_lines
    
    new_df.to_csv("%s.csv" % "kickoffTest", index=False)
    
    return

def testRegex():
    
    df = pd.read_csv("%s.csv" % "kickoffTest")
    
    lines = df['line'].values
    
    for line in lines:
        r0 = re.findall(r"([A-Z][a-z]+|[0-9][0-9]ers) to receive the opening kickoff", line)
        print(line)
        print(r0[0])
        print()
    
    return

#####################

# buildTestData()

testRegex()