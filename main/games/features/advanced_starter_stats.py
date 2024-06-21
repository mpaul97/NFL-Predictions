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
        self.cols = [(self.position + '_advanced_starter_stats_' + col) for col in self.all_pos_cols+self.pos_cols[self.position]]
        self.cols = [(prefix + col) for prefix in ['home_', 'away_'] for col in self.cols]
        return
    def saveFrame(self, df: pd.DataFrame, name):
        df.to_csv("%s.csv" % (self._dir + name), index=False)
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
    def build(self, source: pd.DataFrame, isNew: bool = False):
        def func(row: pd.Series):
            key, home_abbr, away_abbr, wy = row[['key', 'home_abbr', 'away_abbr', 'wy']]
            return list(self.getStats(key, home_abbr, wy)) + list(self.getStats(key, away_abbr, wy))
        fn = self.position + "_advanced_starter_stats"
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        source[self.cols] = source.apply(lambda row: func(row), result_type="expand", axis=1)
        source.fillna(source.mean(), inplace=True)
        if source.isnull().values.any(): # still has nan values
            source.fillna(0, inplace=True)
        fn: str = fn if not isNew else (fn + "_new")
        self.saveFrame(source, fn)
        return source
    
# END / AdvancedStarterStats

##############################

# # sdf = pd.read_csv("%s.csv" % "../../../starters/allStarters")
# sdf = pd.read_csv("%s.csv" % "../../../data/starters_24/starters_w1")
# adf = pd.read_csv("%s.csv" % "../../../data/advancedStats")
# ass = AdvancedStarterStats('qb', sdf, adf, "data/")

# # source = pd.read_csv("%s.csv" % "data/source")
# # ass.build(source)
# source = pd.read_csv("%s.csv" % "data/source_new")
# ass.build(source, True)