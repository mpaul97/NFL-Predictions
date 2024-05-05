import pandas as pd
import numpy as np
import os
import datetime

pd.options.mode.chained_assignment = None

class CoachWinPctPerMonth:
    def __init__(self, df: pd.DataFrame, cdf: pd.DataFrame, _dir: str):
        self.df = df
        self.cdf = cdf[['year', 'abbr', 'coach']]
        self._dir = _dir
        self.target_cols = [
            'key', 'wy', 'home_abbr', 
            'away_abbr', 'home_points', 'away_points', 
            'month'
        ]
        return
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def build(self, source: pd.DataFrame, isNew: bool):
        fn = 'coachWinPctPerMonth'
        if ("%s.csv" % fn) in os.listdir(self._dir) and not isNew:
            print(fn + " already exists.")
            return
        print('Creating ' + fn + '...')
        cd = self.df[self.target_cols]
        cd['home_won'] = cd.apply(lambda x: x['home_points'] >= x['away_points'], axis=1)
        home_cd = cd[['key', 'wy', 'home_abbr', 'month', 'home_won']].rename(columns={'home_abbr': 'abbr', 'home_won': 'won'})
        home_cd['won'] = home_cd['won'].astype(int)
        away_cd = cd[['key', 'wy', 'away_abbr', 'month', 'home_won']].rename(columns={'away_abbr': 'abbr', 'home_won': 'won'})
        away_cd['won'] = away_cd['won'].apply(lambda x: 0 if x else 1)
        cd = pd.concat([home_cd, away_cd]).sort_values(by=['key'])
        cd = self.addDatetimeColumns(cd)
        cd = cd.merge(self.cdf, on=['year', 'abbr'], how='left')
        source = self.addDatetimeColumns(source)
        home_cdf = self.cdf.rename(columns={'abbr': 'home_abbr', 'coach': 'home_coach'})
        source = source.merge(home_cdf, on=['year', 'home_abbr'], how='left')
        away_cdf = self.cdf.rename(columns={'abbr': 'away_abbr', 'coach': 'away_coach'})
        source = source.merge(away_cdf, on=['year', 'away_abbr'], how='left')
        if source['wy'].values[-1] in self.df['wy'].values:
            source = source.merge(self.df[['key', 'month']], on=['key'])
        else:
            source['month'] = source['key'].apply(lambda x: int(x[4:6]))
        new_df = pd.DataFrame(columns=list(source.columns)+['home_coachWinPctPerMonth', 'away_coachWinPctPerMonth'])
        for index, row in source.iterrows():
            self.print_progress_bar(index, len(source.index), fn)
            dt, home_coach, away_coach, month = row[['datetime', 'home_coach', 'away_coach', 'month']]
            home_coach, away_coach = home_coach.split("|")[0], away_coach.split("|")[0]
            home_stats = cd.loc[(cd['datetime']<dt)&(cd['month']==month)&(cd['coach'].str.contains(home_coach)), 'won'].values
            home_stats = np.mean(home_stats) if len(home_stats) != 0 else np.nan
            away_stats = cd.loc[(cd['datetime']<dt)&(cd['month']==month)&(cd['coach'].str.contains(away_coach)), 'won'].values
            away_stats = np.mean(away_stats) if len(away_stats) != 0 else np.nan
            new_df.loc[len(new_df.index)] = list(row.values) + [home_stats, away_stats]
        new_df.drop(columns=['week', 'year', 'datetime', 'home_coach', 'away_coach', 'month'], inplace=True)
        # new_df.fillna(new_df.mean(), inplace=True)
        new_df.fillna(0.5, inplace=True)
        if not isNew:
            self.save_frame(new_df, (self._dir + fn))
        return new_df
    def save_frame(self, df: pd.DataFrame, name):
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

# END / CoachWinPctPerMonth

#############################

# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")
# cdf = pd.read_csv("%s.csv" % "../../../../coaches/coachInfo")
# source = pd.read_csv("%s.csv" % "../source/source")

# cwpm = CoachWinPctPerMonth(cd, cdf, "./")
# # cwpm.build(source, False)

# new_source = pd.read_csv("%s.csv" % "../source/new_source")
# df = cwpm.build(new_source, True)
# print(df)
        