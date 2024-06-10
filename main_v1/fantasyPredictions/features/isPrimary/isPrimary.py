import pandas as pd
import numpy as np
import os
import regex as re

pd.options.mode.chained_assignment = None

class IsPrimary:
    def __init__(self, _dir):
        self._dir = _dir
        self.positions = ['QB', 'RB', 'WR', 'TE']
        return
    def buildIsPrimary(self, source: pd.DataFrame, cd: pd.DataFrame):
        if 'isPrimary.csv' in os.listdir(self._dir):
            print('isPrimary already exists.')
            return
        is_primary = []
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), 'isPrimary')
            key, abbr, pid, position = row[['key', 'abbr', 'p_id', 'position']]
            vols = cd.loc[
                (cd['game_key']==key)&
                (cd['abbr']==abbr)&
                (cd['position']==position), 
                ['p_id', 'volume_percentage']
            ].values
            vols = vols[vols[:, 1].argsort()[::-1]]
            primary = 1 if vols[0][0] == pid else 0
            is_primary.append(primary)
        source['isPrimary'] = is_primary
        self.saveFrame(source, (self._dir + 'isPrimary'))
        return
    def buildNewIsPrimary(self, source: pd.DataFrame, cd: pd.DataFrame):
        """
        Builds new isPrimary, finds best player by position in starters
        Args:
            source (pd.DataFrame): uses current week starters
            cd (pd.DataFrame): position data
        """
        print('Creating new isPrimary...')
        new_df = pd.DataFrame(columns=['p_id', 'isPrimary'])
        wy = source['wy'].values[0]
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        for abbr in set(source['abbr']):
            for position in self.positions:
                players = source.loc[(source['abbr']==abbr)&(source['position']==position), 'p_id'].values
                if len(players) > 1:
                    info = []
                    target_year = str(year - 1) if week == 1 else str(year)
                    for p in players:
                        vols = sum(cd.loc[(cd['p_id']==p)&(cd['wy'].str.contains(target_year)), 'volume_percentage'].values)
                        info.append((p, vols))
                    info.sort(key=lambda x: x[1], reverse=True)
                    for p, _ in info:
                        new_df.loc[len(new_df.index)] = [p, (1 if info[0][0] == p else 0)]
                else:
                    try:
                        new_df.loc[len(new_df.index)] = [players[0], 1]
                    except IndexError:
                        print(f"No players found for {abbr}, {position}")
        source = source.merge(new_df, on=['p_id'])
        return source
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
        return
    
###########################

# ONLY #1 Players

# ip = IsPrimary("./")

# source = pd.read_csv("%s.csv" % "../source/source")
# POSITION_PATH = "../../../../data/positionData/"
# fns = [fn for fn in os.listdir(POSITION_PATH) if re.search(r"(QB|RB|WR|TE)[A-Z][a-z]+", fn)]
# ocd = pd.concat([pd.read_csv(POSITION_PATH + fn) for fn in fns])

# ip.buildIsPrimary(source, ocd)

# source = pd.read_csv("%s.csv" % "../source/new_source")
# POSITION_PATH = "../../../../data/positionData/"
# fns = [fn for fn in os.listdir(POSITION_PATH) if re.search(r"(QB|RB|WR|TE)[A-Z][a-z]+", fn)]
# ocd = pd.concat([pd.read_csv(POSITION_PATH + fn) for fn in fns])

# ip = IsPrimary("./")
# ip.buildNewIsPrimary(source, ocd)