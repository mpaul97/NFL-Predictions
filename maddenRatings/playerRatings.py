import pandas as pd
import numpy as np
import os
import datetime

class PlayerRatings:
    def __init__(self, _dir):
        self._dir = _dir
        self.position_dir = self._dir + "../data/positionData/"
        self.sdf = pd.read_csv("%s.csv" % (self._dir + "../starters/allStarters"))
        self.sdf: pd.DataFrame = self.addDatetimeColumns(self.sdf)
        self.fpi = pd.read_csv("%s.csv" % (self._dir + "../playerNames/finalPlayerInfo"))
        self.df = pd.read_csv("%s.csv" % (self._dir + "allOverallRatings_01-23"))
        self.df1 = pd.read_csv("%s.csv" % (self._dir + "pred_ratings/predOverallRatings_94-00"))
        self.cd: pd.DataFrame = None
        return
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def buildPlayerRatings(self):
        self.setCd()
        cd = self.cd
        new_df = pd.DataFrame(columns=['year', 'abbr', 'p_id', 'position', 'rating'])
        for year in range(1994, 2023):
            print(year)
            abbrs = list(set(self.sdf.loc[self.sdf['year']==year, 'abbr']))
            for abbr in abbrs:
                sdf = self.sdf.loc[(self.sdf['abbr']==abbr)&(self.sdf['year']==year)]
                all_starters = list(set(('|'.join(sdf['starters'].values)).split("|")))
                players = [(s.split(":")[0], s.split(":")[1]) for s in all_starters]
                others = cd.loc[(cd['abbr']==abbr)&(cd['wy'].str.contains(str(year))), ['p_id', 'position']].values
                [players.append((pid, position)) for pid, position in others if (pid, position) not in players]
                df = self.df if year >= 2001 else self.df1
                for pid, position in players:
                    try:
                        rating = df.loc[(df['p_id']==pid)&(df['year']==year), 'overall_rating'].values[0]
                    except IndexError:
                        rating = np.nan
                    new_df.loc[len(new_df.index)] = [year, abbr, pid, position, rating]
        new_df.fillna(new_df.mean(), inplace=True)
        new_df = new_df.round(0)
        self.saveFrame(new_df, (self._dir + "playerRatings"))
        return
    def addNewRatings(self, year: int):
        df = self.df
        new_df = pd.read_csv("%s.csv" % (self._dir + "playerRatings"))
        if year not in new_df['year'].values:
            df = df.loc[(df['year']==year)&(df['p_id']!='UNK')&(df['position']!='UNK_POS')]
            df = df[['year', 'abbr', 'p_id', 'position', 'overall_rating']]
            df.columns = new_df.columns
            self.saveFrame(pd.concat([new_df, df]), (self._dir + "playerRatings"))
        else:
            print("Year already included.")
        return
    def updateRatings(self):
        """
        Add player with missing positions to ratings
        """
        df = self.df
        unk_df = df.loc[df['position']=='UNK_POS']
        for index, row in unk_df.iterrows():
            pid = row['p_id']
            try:
                position = self.fpi.loc[self.fpi['p_id']==pid, 'position'].values[0]
                if position != 'UNK':
                    df.at[index, 'position'] = position
            except IndexError:
                continue
        self.saveFrame(df, 'temp')
        return
    def setCd(self):
        self.cd = pd.concat([pd.read_csv(self.position_dir + fn) for fn in os.listdir(self.position_dir) if '.csv' in fn])
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
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
    
# END / PlayerRatings

#################################

pr = PlayerRatings("./")

# pr.buildPlayerRatings()

pr.addNewRatings(2023)

# pr.updateRatings()