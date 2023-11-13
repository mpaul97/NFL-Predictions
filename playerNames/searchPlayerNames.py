import pandas as pd
import numpy as np

def search():
    
    df = pd.read_csv("%s.csv" % "playerNames")
    
    searching = True
    
    while searching:
        val = input("Enter a player name (\'q\' to exit):")
        if val == 'q':
            searching = False
        else:
            temp = df.loc[df['name'].str.contains(val, na=False, case=False)]
            for index, row in temp.iterrows():
                print(row['p_id'] + " : " + row['name'])
            
#############################

search()