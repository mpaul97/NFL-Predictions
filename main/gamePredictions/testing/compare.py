import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt

def getInfo(df: pd.DataFrame, abbr, year):
    return df.loc[(df['abbr']==abbr)&(df['wy'].str.contains(str(year))), ['elo', 'wy']].values

def compareElos():
    teamElos = pd.read_csv("%s.csv" % "../features/teamElos/rawElos")
    coachElos = pd.read_csv("%s.csv" % "../features/coachElos/rawCoachElos")
    abbrs = ['TAM', 'NYG', 'JAX', 'GNB']
    year = 2022
    rows, cols = 2, 2
    fig, ax = plt.subplots(rows, cols)
    # 0,0 0,1, 1,0, 1,1
    for index, abbr in enumerate(abbrs):
        x = index // rows
        y = index % cols
        t_info = getInfo(teamElos, abbr, year)
        c_info = getInfo(coachElos, abbr, year)
        ax[x, y].plot(t_info[:, 0])
        ax[x, y].plot(c_info[:, 0])
        ax[x, y].set_title(abbr)
        ax[x, y].legend(['team', 'coach'])
    plt.show()
    return

########################

compareElos()