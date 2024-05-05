import pandas as pd
import numpy as np
import os

class OlStatsData:
    def __init__(self, _dir):
        self._dir = _dir
        self.position_dir = self._dir + "positionData/"
        self.stats_dir = self.position_dir + "OLStatsData/"
        self.starters_dir = self._dir + "../starters/"
        # frames
        self.sdf = pd.read_csv("%s.csv" % (self.starters_dir + "allStarters"))
        self.data = pd.concat([pd.read_csv("%s.csv" % (self.position_dir + "RBData")), pd.read_csv("%s.csv" % (self.position_dir + "QBData"))])
        return
    def zeroDivision(self, n, d):
        return n / d if d else 0
    def getOlStarters(self, starters: list[str]):
        return [(s.replace(":OL","")) for s in starters if 'OL' in s]
    def build(self):
        data = self.data
        new_df = pd.DataFrame(columns=['p_id', 'game_key', 'abbr', 'wy', 'times_sacked',
                                    'yards_lost_from_sacks', 'sack_percentage',
                                    'rush_yards_per_attempt'])
        targetStatsQbs = ['times_sacked', 'yards_lost_from_sacks', 'sack_percentage']
        targetStatsAll = ['rush_yards_per_attempt']
        for index, row in self.sdf.iterrows():
            self.print_progress_bar(index, len(self.sdf.index), 'Updating OLStatsData')
            wy = row['wy']
            abbr, key = row[['abbr', 'key']]
            players = row['starters'].split("|")
            starters = self.getOlStarters(players)
            stats_qb = data.loc[(data['game_key']==key)&(data['abbr']==abbr), targetStatsQbs]
            stats_qb.dropna(axis=0, inplace=True)
            qbLen = len(stats_qb.index)
            stats_qb = stats_qb.sum()
            stats_qb = stats_qb.apply(lambda x: self.zeroDivision(x, qbLen))
            stats_all = data.loc[(data['game_key']==key)&(data['abbr']==abbr), targetStatsAll]
            stats_all = stats_all.loc[~(stats_all==0).all(axis=1)]
            allLen = len(stats_all.index)
            stats_all = stats_all.sum()
            stats_all = stats_all.apply(lambda x: self.zeroDivision(x, allLen))
            for s in starters:
                ts = stats_qb[targetStatsQbs[0]]
                ylfs = stats_qb[targetStatsQbs[1]]
                sp = stats_qb[targetStatsQbs[2]]
                rypa = stats_all[targetStatsAll[0]]
                new_df.loc[len(new_df.index)] = [s, key, abbr, wy, ts, ylfs, sp, rypa]
        self.save_frame(new_df, (self.stats_dir + "OLStatsData"))
        return
    def update(self):
        if 'OLStatsData.csv' not in os.listdir(self.stats_dir):
            print("OLStatsData.csv not found.")
            return
        df = pd.read_csv("%s.csv" % (self.stats_dir + "OLStatsData"))
        missing_wys = set(self.sdf['wy']).difference(set(df['wy']))
        self.sdf = self.sdf.loc[self.sdf['wy'].isin(missing_wys)]
        self.sdf = self.sdf.reset_index(drop=True)
        if len(missing_wys) == 0:
            print("OLStatsData.csv up-to-date.")
            return
        print(f"Updating OLStatsData for length of wys: {len(missing_wys)}...")
        data = self.data
        new_df = pd.DataFrame(columns=['p_id', 'game_key', 'abbr', 'wy', 'times_sacked',
                                    'yards_lost_from_sacks', 'sack_percentage',
                                    'rush_yards_per_attempt'])
        targetStatsQbs = ['times_sacked', 'yards_lost_from_sacks', 'sack_percentage']
        targetStatsAll = ['rush_yards_per_attempt']
        for index, row in self.sdf.iterrows():
            self.print_progress_bar(index, len(self.sdf.index), 'Updating OLStatsData')
            wy = row['wy']
            abbr, key = row[['abbr', 'key']]
            players = row['starters'].split("|")
            starters = self.getOlStarters(players)
            stats_qb = data.loc[(data['game_key']==key)&(data['abbr']==abbr), targetStatsQbs]
            stats_qb.dropna(axis=0, inplace=True)
            qbLen = len(stats_qb.index)
            stats_qb = stats_qb.sum()
            stats_qb = stats_qb.apply(lambda x: self.zeroDivision(x, qbLen))
            stats_all = data.loc[(data['game_key']==key)&(data['abbr']==abbr), targetStatsAll]
            stats_all = stats_all.loc[~(stats_all==0).all(axis=1)]
            allLen = len(stats_all.index)
            stats_all = stats_all.sum()
            stats_all = stats_all.apply(lambda x: self.zeroDivision(x, allLen))
            for s in starters:
                ts = stats_qb[targetStatsQbs[0]]
                ylfs = stats_qb[targetStatsQbs[1]]
                sp = stats_qb[targetStatsQbs[2]]
                rypa = stats_all[targetStatsAll[0]]
                new_df.loc[len(new_df.index)] = [s, key, abbr, wy, ts, ylfs, sp, rypa]
        new_df = pd.concat([df, new_df])
        self.save_frame(new_df, (self.stats_dir + "OLStatsData"))
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def print_progress_bar(self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
# END / OlStatsData

########################

# osd = OlStatsData("./")

# osd.update()