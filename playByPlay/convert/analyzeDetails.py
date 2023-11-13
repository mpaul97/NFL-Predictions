import pandas as pd
import numpy as np
import os
import regex as re

def analyze():
    
    with open('rawTrain.txt', 'r') as f:
        lines = f.readlines()
        
    # lines = lines[:1000]
    
    dif_lines = []
    
    for index, line in enumerate(lines):
        print(index, len(lines))
        line = line.strip()
        if not re.match(r"^[A-Z][a-z]", line):
            dif_lines.append(line)
            
    new_file = open('difLines.txt', 'w')
    
    dif_lines = list(set(dif_lines))
    
    for line in dif_lines:
        new_file.write(line + "\n")
    
    new_file.close()
    
    return

##################

analyze()