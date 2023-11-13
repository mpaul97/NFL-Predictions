import pandas as pd
import numpy as np

def cleanQb():

    df = pd.read_csv("%s.csv" % "QBData")

    cols = list(df.columns)

    startCol = 'times_pass_target'
    endCol ='longest_punt'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % "QBData", index=False)

def cleanRb():

    df = pd.read_csv("%s.csv" % "RBData")

    cols = list(df.columns)

    startCol = 'completed_passes'
    endCol ='quarterback_rating'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    startCol1 = 'interceptions'
    endCol1 = 'longest_punt'

    startIndex1 = cols.index(startCol1)
    endIndex1 = cols.index(endCol1) + 1

    drop_cols += cols[startIndex1:endIndex1]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % "RBData", index=False)

def cleanWr():

    df = pd.read_csv("%s.csv" % "WRData")

    cols = list(df.columns)

    startCol = 'completed_passes'
    endCol ='quarterback_rating'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    startCol1 = 'interceptions'
    endCol1 = 'longest_punt'

    startIndex1 = cols.index(startCol1)
    endIndex1 = cols.index(endCol1) + 1

    drop_cols += cols[startIndex1:endIndex1]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % "WRData", index=False)

def cleanTe():

    df = pd.read_csv("%s.csv" % "TEData")

    cols = list(df.columns)

    startCol = 'completed_passes'
    endCol ='quarterback_rating'

    startIndex = cols.index(startCol)
    endIndex = cols.index(endCol) + 1

    drop_cols = cols[startIndex:endIndex]

    startCol1 = 'interceptions'
    endCol1 = 'longest_punt'

    startIndex1 = cols.index(startCol1)
    endIndex1 = cols.index(endCol1) + 1

    drop_cols += cols[startIndex1:endIndex1]

    df.drop(columns=drop_cols, inplace=True)

    df.to_csv("%s.csv" % "TEData-1", index=False)

def cleanLBDL():
    """
    Remove unnessary columns
    """
    df = pd.read_csv("%s.csv" % "LBDLData")
    cols = [
        'p_id', 'interceptions', 'yards_returned_from_interception',
        'interceptions_returned_for_touchdown', 'longest_interception_return',
        'passes_defended', 'sacks', 'combined_tackles', 'solo_tackles',
        'assists_on_tackles', 'tackles_for_loss', 'quarterback_hits',
        'fumbles_recovered', 'yards_recovered_from_fumble',
        'fumbles_recovered_for_touchdown', 'fumbles_forced',
        'kickoff_returns', 'kickoff_return_yards', 'average_kickoff_return_yards',
        'kickoff_return_touchdown', 'longest_kickoff_return', 'punt_returns',
        'punt_return_yards', 'yards_per_punt_return', 'punt_return_touchdown',
        'abbr', 'game_key', 'isHome', 'wy', 'position'
    ]
    df = df[cols]
    df.to_csv("%s.csv" % "LBDLData", index=False)
    return

################################

# cleanQb()

# cleanRb()

# cleanWr()

# cleanTe()

cleanLBDL()