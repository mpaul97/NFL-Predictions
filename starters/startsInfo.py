import pandas as pd
import numpy as np
import os
import random
import datetime
import time

class StartsInfo:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "../data/"
        self.position_dir = self.data_dir + "positionData/"
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        # frames
        self.sdf: pd.DataFrame = pd.read_csv("%s.csv" % (self._dir + "allStarters"))
        self.sdf: pd.DataFrame = self.addDatetimeColumns(self.sdf)
        self.gd: pd.DataFrame = pd.read_csv("%s.csv" % (self.data_dir + "gameData"))
        self.gd: pd.DataFrame = self.addDatetimeColumns(self.gd)
        self.position_frames: dict = {
            pos: (pd.read_csv("%s.csv" % (self.position_dir + pos + "Data")) if pos not in ['LB', 'DL'] else pd.read_csv("%s.csv" % (self.position_dir + "LBDLData")))
            for pos in self.positions
        }
        self.all_pids: pd.DataFrame = None
        # concatenate all position frames, remove duplicates, sort by datetime
        self.cd = pd.concat(self.position_frames.values())
        self.cd.drop_duplicates(inplace=True)
        self.cd = self.addDatetimeColumns(self.cd)
        self.cd.reset_index(drop=True, inplace=True)
        self.cd.sort_values(by=['datetime'], inplace=True)
        # starts info
        self.si: pd.DataFrame = None
        # time
        self.start: float = 0
        self.end: float = 0
        self.elapsed: float = 0
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def createAllPids(self):
        """
        Create dataframe containing all pids + position in starters \n
        and position data
        """
        df = pd.concat(self.position_frames.values())
        df = df[['p_id', 'position']]
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        # starters
        all_starters = [(p.split(":")[0], p.split(":")[1]) for p in ('|'.join(self.sdf['starters'].values)).split("|")]
        new_df = pd.DataFrame(all_starters, columns=['p_id', 'position'])
        df = pd.concat([df, new_df])
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        self.saveFrame(df, 'allPids')
        return
    def startersToFrame(self, starters: str):
        df = pd.DataFrame(columns=['p_id', 'position'])
        for s in starters.split("|"):
            df.loc[len(df.index)] = [s.split(":")[0], s.split(":")[1]]
        return df
    def createStartsInfo(self):
        gd, sdf, cd = self.gd, self.sdf, self.cd
        _types = ['career', 'curr_season', 'last_season']
        cols = [(_type + "_" + suffix) for suffix in ['gamesPlayed', 'starts'] for _type in _types]
        new_df = pd.DataFrame(columns=['key', 'p_id', 'position', 'wy']+cols)
        for index, (key, week, year, datetime) in enumerate(gd[['key', 'week', 'year', 'datetime']].values):
            self.printProgressBar(index, len(gd.index), 'Starts Info')
            players = cd.loc[cd['game_key']==key, ['p_id', 'position']]
            starters = sdf.loc[sdf['key']==key, 'starters'].values[0]
            starters = self.startersToFrame(starters)
            all_df = pd.concat([players, starters])
            all_df.drop_duplicates(inplace=True)
            all_games = cd.loc[cd['datetime']<datetime, 'p_id'].values
            curr_season_games = cd.loc[(cd['datetime']<datetime)&(cd['year']==year), 'p_id'].values
            last_season_games = cd.loc[cd['year']==(year-1), 'p_id'].values
            all_starters = '|'.join(sdf.loc[sdf['datetime']<datetime, 'starters'].values)
            curr_season_starters = '|'.join(sdf.loc[(sdf['datetime']<datetime)&(sdf['year']==year), 'starters'].values)
            last_season_starters = '|'.join(sdf.loc[sdf['year']==(year-1), 'starters'].values)
            for pid, position in all_df[['p_id', 'position']].values:
                career_gp = len(all_games[all_games==pid])
                curr_season_gp = len(curr_season_games[curr_season_games==pid])
                last_season_gp = len(last_season_games[last_season_games==pid])
                career_starts = all_starters.count(pid)
                curr_season_starts = curr_season_starters.count(pid)
                last_season_starts = last_season_starters.count(pid)
                new_df.loc[len(new_df.index)] = [
                    key, pid, position, (str(week) + " | " + str(year)),
                    career_gp, curr_season_gp, last_season_gp,
                    career_starts, curr_season_starts, last_season_starts
                ]
        self.saveFrame(new_df, 'startsInfo')
        return
    def updateStartsInfo(self):
        self.setSi()
        try:
            missing_wy = list(set(self.gd['wy']).difference(set(self.si['wy'])))[0]
            print(f"Updating startsInfo for wy: {missing_wy}")
            gd = self.gd.loc[self.gd['wy']==missing_wy]
            sdf, cd = self.sdf, self.cd
            _types = ['career', 'curr_season', 'last_season']
            cols = [(_type + "_" + suffix) for suffix in ['gamesPlayed', 'starts'] for _type in _types]
            new_df = pd.DataFrame(columns=['key', 'p_id', 'position', 'wy']+cols)
            for index, (key, week, year, datetime) in enumerate(gd[['key', 'week', 'year', 'datetime']].values):
                players = cd.loc[cd['game_key']==key, ['p_id', 'position']]
                starters = sdf.loc[sdf['key']==key, 'starters'].values[0]
                starters = self.startersToFrame(starters)
                all_df = pd.concat([players, starters])
                all_df.drop_duplicates(inplace=True)
                all_games = cd.loc[cd['datetime']<datetime, 'p_id'].values
                curr_season_games = cd.loc[(cd['datetime']<datetime)&(cd['year']==year), 'p_id'].values
                last_season_games = cd.loc[cd['year']==(year-1), 'p_id'].values
                all_starters = '|'.join(sdf.loc[sdf['datetime']<datetime, 'starters'].values)
                curr_season_starters = '|'.join(sdf.loc[(sdf['datetime']<datetime)&(sdf['year']==year), 'starters'].values)
                last_season_starters = '|'.join(sdf.loc[sdf['year']==(year-1), 'starters'].values)
                for pid, position in all_df[['p_id', 'position']].values:
                    career_gp = len(all_games[all_games==pid])
                    curr_season_gp = len(curr_season_games[curr_season_games==pid])
                    last_season_gp = len(last_season_games[last_season_games==pid])
                    career_starts = all_starters.count(pid)
                    curr_season_starts = curr_season_starters.count(pid)
                    last_season_starts = last_season_starters.count(pid)
                    new_df.loc[len(new_df.index)] = [
                        key, pid, position, (str(week) + " | " + str(year)),
                        career_gp, curr_season_gp, last_season_gp,
                        career_starts, curr_season_starts, last_season_starts
                    ]
            si = pd.concat([self.si, new_df])
            self.saveFrame(si, "startsInfo")
        except IndexError:
            print("startsInfo already up-to-date.")
        return
    def setSi(self):
        self.si = pd.read_csv("%s.csv" % (self._dir + "startsInfo"))
        return
    def setAllPids(self):
        self.all_pids = pd.read_csv("%s.csv" % (self._dir + "allPids"))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % (self._dir + name), index=False)
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
    
# / END StartsInfo

################################

# start = time.time()

# a = StartsInfo('./')

# # a.createAllPids()

# # a.createStartsInfo()

# a.updateStartsInfo()

# end = time.time()
# elapsed = end - start
# print(f"Time Elapsed: {elapsed}")