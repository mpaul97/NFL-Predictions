import pandas as pd
import numpy as np

def saveAllPositions():
    df = pd.read_csv("%s.csv" % "allStarters")
    all_starters = "|".join(df['starters'].values)
    poses = list(set([s[-2:] for s in all_starters.split("|")]))
    new_df = pd.DataFrame()
    new_df['position'] = poses
    new_df.to_csv("%s.csv" % "allPositions", index=False)
    return

#########################

saveAllPositions()