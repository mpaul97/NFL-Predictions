import pandas as pd
import numpy as np
import os

def get_player_info_positions_freq():
    df = pd.read_csv("%s.csv" % "../playerNames_v2/data/playerInfo")
    positions = df['positions'].value_counts()
    for pos, count in positions.items():
        print(f"position: {pos} - count: {count}")
    return

#######################

get_player_info_positions_freq()