import pandas as pd
import numpy as np
import os

from sportsipy.nfl.roster import Player

class AdvancedStarterStats:
    def __init__(self, position: str, sdf: pd.DataFrame, adf: pd.DataFrame, _dir):
        self.position = position
        self.sdf = sdf
        self.adf = adf
        self._dir = _dir
        # info
        self.all_pos_cols = [
            'approximate_value', 'games','games_started',
            'fumbles'
        ]
        self.pos_cols = {
            'qb': [
                'adjusted_net_yards_per_attempt_index', 'adjusted_net_yards_per_pass_attempt',
                'adjusted_yards_per_attempt', 'adjusted_yards_per_attempt_index',
                'attempted_passes', 'completed_passes', 'completion_percentage_index',
                'espn_qbr', 'fourth_quarter_comebacks', 'game_winning_drives',
                'interception_percentage', 'interception_percentage_index',
                'interceptions_thrown', 'net_yards_per_attempt_index',
                'net_yards_per_pass_attempt', 'passer_rating_index',
                'passing_completion', 'passing_touchdown_percentage',
                'passing_touchdowns', 'passing_yards', 'passing_yards_per_attempt',
                'quarterback_rating', 'sack_percentage_index', 'times_sacked',
                'touchdown_percentage_index', 'yards_lost_to_sacks', 'yards_per_attempt_index',
                'yards_per_completed_pass', 'yards_per_game_played'
            ]
        }
        return
    def writeAllColumns(self):
        pid = 'RodgAa00'
        p = Player(pid)
        df = p.dataframe
        file = open((self._dir + "columns.txt"), "w")
        file.write('\n'.join(list(df.columns)))
        file.close()
        return
    def convertRecord(self, record: str):
        if type(record) == float:
            return 0
        wins, loses, ties = record.split("-")
        wins, loses, ties = int(wins), int(loses), int(ties)
        return (2*(wins+ties))/(2*sum([wins, loses, ties]))
    def getStats(self, key: str, abbr: str, wy: str):
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        starters = (self.sdf.loc[(self.sdf['key']==key)&(self.sdf['abbr']==abbr), 'starters'].values[0]).split("|")
        try:
            pid = [p.split(":")[0] for p in starters if self.position.upper() in p][0]
            df = self.adf.loc[self.adf['player_id']==pid]
            if not df.empty:
                season = str(year-1) if week == 1 else str(year)
                stats: pd.DataFrame = df.loc[df['season']==season, self.all_pos_cols+self.pos_cols[self.position]]
                if not stats.empty:
                    if stats['games'].values[0] < 2:
                        stats.fillna(0, inplace=True)
                    # if self.position == 'qb': # convert qb_record
                    #     record = stats['qb_record'].values[0]
                    #     wlp = self.convertRecord(record)
                    #     stats.at[stats.index[0], 'qb_record'] = wlp
                    stats = stats.values[0]
                else:
                    stats = np.zeros(len(self.all_pos_cols+self.pos_cols[self.position]))
            else:
                stats = np.zeros(len(self.all_pos_cols+self.pos_cols[self.position]))
        except IndexError:
            stats = np.zeros(len(self.all_pos_cols+self.pos_cols[self.position]))
        return stats
    def buildAdvancedStarterStats(self, source: pd.DataFrame, isNew: bool):
        fn = self.position + "_advancedStarterStats"
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        cols = [(self.position + '_advancedStarterStats_' + col) for col in self.all_pos_cols+self.pos_cols[self.position]]
        cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in cols]
        new_df = pd.DataFrame(columns=list(source.columns)+cols)
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), fn)
            key = row['key']
            home_abbr, away_abbr = row[['home_abbr', 'away_abbr']]
            wy = row['wy']
            home_stats = self.getStats(key, home_abbr, wy)
            away_stats = self.getStats(key, away_abbr, wy)
            new_df.loc[len(new_df.index)] = list(row.values) + list(home_stats) + list(away_stats)
        new_df.fillna(new_df.mean(), inplace=True)
        if not isNew:
            self.saveFrame(new_df, (self._dir + fn))
        if new_df.isnull().values.any(): # still has nan values
            new_df.fillna(0, inplace=True)
        return new_df
    def saveFrame(self, df: pd.DataFrame, name):
        df.to_csv("%s.csv" % name, index=False)
        return
    def printProgressBar(self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
    
# END / AdvancedStarterStats

##############################

# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")
# adf = pd.read_csv("%s.csv" % "../../../../data/advancedStats")
# ass = AdvancedStarterStats('qb', sdf, adf, "./")

# # # source = pd.read_csv("%s.csv" % "../source/source")
# source = pd.read_csv("%s.csv" % "../source/new_source")
# df = ass.buildAdvancedStarterStats(source, True)
# ass.saveFrame(df, "temp")