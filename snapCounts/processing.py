import pandas as pd
import numpy as np
import os
import time
import datetime
import multiprocessing

class Processing:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.main_dir = self._dir + "../data/"
        self.names_dir = self._dir + "../playerNames_v2/data/"
        self.starters_dir = self._dir + "../starters/"
        self.sdf = pd.read_csv("%s.csv" % (self._dir + "snap_counts"))
        self.sl = pd.read_csv("%s.csv" % (self.main_dir + "seasonLength"))
        self.sp = pd.read_csv("%s.csv" % (self.data_dir + "snap_positions"))
        self.teams = pd.read_csv("%s.csv" % (self.names_dir + "playerTeams"))
        self.starters = pd.read_csv("%s.csv" % (self.starters_dir + "allStarters"))
        self.st_positions = ['K', 'P', 'LS']
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def get_position(self, position: str):
        return self.sp.loc[self.sp['position']==position, 'simplePosition'].values[0]
    def significant_injuries_func(self, pids: list[str]):
        """
        No snaps in next game and
        current snap percentage greater than 0.2
        """
        new_df = pd.DataFrame(columns=['key', 'wy', 'abbr', 'p_id', 'position', 'significant_injury'])
        for pid in pids:
            df: pd.DataFrame = self.sdf.loc[self.sdf['p_id']==pid]
            df = df.reset_index(drop=True)
            for index, row in df.iterrows():
                if index != 0:
                    key, position, abbr, wy, dt, year = row[['key', 'position', 'abbr', 'wy', 'datetime', 'year']]
                    isOff = (self.sp.loc[self.sp['position']==position, 'isOff'].values[0] == 1)
                    pct_col = 'off_pct' if isOff else 'def_pct'
                    curr_pct = row[pct_col]
                    try:
                        next_wy = self.sdf.loc[(self.sdf['datetime']>dt)&(self.sdf['abbr']==abbr), 'wy'].values[0]
                        next_week, next_year = [int(w) for w in next_wy.split(" | ")]
                        season_length = self.sl.loc[self.sl['year']==next_year, 'weeks'].values[0]
                        next_pct = df.loc[df['wy']==next_wy, pct_col].values
                        next_pct = next_pct[0] if len(next_pct) != 0 else 0
                        # total_mean = np.mean(self.sdf.loc[
                        #     (self.sdf['year']==year)&
                        #     (self.sdf['position']==position),
                        #     pct_col
                        # ].values)
                        starters = self.starters.loc[(self.starters['key']==key)&(self.starters['abbr']==abbr), 'starters'].values[0]
                        if next_pct == 0 and (curr_pct > 0.2 or pid in starters) and next_week < season_length:
                            new_df.loc[len(new_df.index)] = [key, wy, abbr, pid, position, 1]
                    except IndexError: # no last data
                        continue
        return new_df
    def significant_injuries_parallel(self):
        self.sdf = self.addDatetimeColumns(self.sdf)
        self.sdf['position'] = self.sdf['position'].apply(lambda x: self.get_position(x))
        self.sdf = self.sdf.loc[~self.sdf['position'].isin(self.st_positions)]
        pids = list(set(self.sdf['p_id']))
        num_cores = multiprocessing.cpu_count()-1
        num_partitions = num_cores
        pids_split = np.array_split(pids, num_partitions)
        df_list = []
        if __name__ == '__main__':
            pool = multiprocessing.Pool(num_cores)
            all_dfs = pd.concat(pool.map(self.significant_injuries_func, pids_split))
            df_list.append(all_dfs)
            pool.close()
            pool.join()
        if df_list:
            new_df = pd.concat(df_list)
            new_df.sort_values(by=['key'], inplace=True)
            self.save_frame(new_df, (self.data_dir + "significant_injuries"))
        return
    def significant_injuries(self):
        """
        No snaps in next game
        """
        new_df = pd.DataFrame(columns=['key', 'wy', 'abbr', 'p_id', 'position', 'significant_injury'])
        self.sdf = self.addDatetimeColumns(self.sdf)
        pids = list(set(self.sdf['p_id']))
        for idx, pid in enumerate(pids):
            df: pd.DataFrame = self.sdf.loc[self.sdf['p_id']==pid]
            df = df.reset_index(drop=True)
            # teams = self.teams.loc[self.teams['p_id']==pid]
            # abbrs = {}
            # for year in range(2012, int(teams.columns[-1])+1):
            #     val = teams[str(year)].values[0]
            #     abbrs[year] = val.split("|") if not pd.isna(val) else []
            for index, row in df.iterrows():
                if index != 0:
                    key, position, abbr, wy, dt, year = row[['key', 'position', 'abbr', 'wy', 'datetime', 'year']]
                    isOff = (self.sp.loc[self.sp['position']==position, 'isOff'].values[0] == 1)
                    pct_col = 'off_pct' if isOff else 'def_pct'
                    curr_pct = row[pct_col]
                    # curr_abbrs = abbrs[year]
                    try:
                        next_wy = self.sdf.loc[(self.sdf['datetime']>dt)&(self.sdf['abbr']==abbr), 'wy'].values[0]
                        next_week, next_year = [int(w) for w in next_wy.split(" | ")]
                        season_length = self.sl.loc[self.sl['year']==next_year, 'weeks'].values[0]
                        next_pct = df.loc[df['wy']==next_wy, pct_col].values
                        next_pct = next_pct[0] if len(next_pct) != 0 else 0
                        if next_pct == 0 and next_week < season_length:
                            new_df.loc[len(new_df.index)] = [key, wy, abbr, pid, position, 1]
                        # next_pct, next_wy = df.loc[df.index==index+1, [pct_col, 'wy']].values[0]
                        # next_week, next_year = [int(w) for w in next_wy.split(" | ")]
                        # season_length = self.sl.loc[self.sl['year']==next_year, 'weeks'].values[0]
                        # if curr_pct - next_pct > 0.5 and next_week < season_length:
                        #     print(curr_pct, next_pct, next_wy)
                    except IndexError: # no last data
                        continue
        new_df.sort_values(by=['key'], inplace=True)
        self.save_frame(new_df, (self.data_dir + "significant_injuries"))
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
    
# END / Processing

######################

p = Processing("./")

p.significant_injuries_parallel()