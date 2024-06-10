import os

root = os.getcwd()

root = root.replace("main","")

POSITION_PATH = root + "/data/positionData/"
DATA_PATH = root + "/data/"
STARTERS_PATH = root + 'starters/'
TEAMNAMES_PATH = root + 'teamNames/'
COACHES_PATH = root + 'coaches/'
MADDEN_PATH = root + 'maddenRatings/'
NAMES_PATH = root + 'playerNames_v2/data/'
SNAP_PATH = root + 'snapCounts/'
PLAYER_RANKS_PATH = root + 'playerRanks/'
PBP_PATH = root + 'playByPlay_v2/data/features/'

ALL_PATHS = {
    'POSITION_PATH': POSITION_PATH, 'DATA_PATH': DATA_PATH, 'STARTERS_PATH': STARTERS_PATH,
    'TEAMNAMES_PATH': TEAMNAMES_PATH, 'COACHES_PATH': COACHES_PATH, 'MADDEN_PATH': MADDEN_PATH,
    'NAMES_PATH': NAMES_PATH, 'SNAP_PATH': SNAP_PATH, 'PLAYER_RANKS_PATH': PLAYER_RANKS_PATH,
    'PBP_PATH': PBP_PATH
}