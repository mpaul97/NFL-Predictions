import pandas as pd
import numpy as np
import os

class StarterMaddenRatings:
    def __init__(self, _dir):
        self._dir = _dir
        self.positions = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB']
        return
    def getInfo(self, wy, abbr, sdf: pd.DataFrame, rdf: pd.DataFrame):
        starters = (sdf.loc[(sdf['wy']==wy)&(sdf['abbr']==abbr), 'starters'].values[0]).split("|")
        starters = [{ 'pid': s.split(":")[0], 'position': s.split(":")[1] } for s in starters]
        year = int(wy.split(" | ")[1])
        for s in starters:
            try:
                s['rating'] = rdf.loc[(rdf['p_id']==s['pid'])&(rdf['year']==year), 'rating'].values[0]
            except IndexError:
                s['rating'] = np.nan
        total_avg = np.mean([s['rating'] for s in starters if not pd.isna(s['rating'])])
        pos_avgs, pos_min, pos_max = [], [], []
        for pos in self.positions:
            pos_ratings = [s['rating'] for s in starters if s['position']==pos]
            pos_ratings = [r for r in pos_ratings if not pd.isna(r)]
            if len(pos_ratings) != 0:
                pos_avgs.append(np.mean(pos_ratings))
                pos_min.append(min(pos_ratings))
                pos_max.append(max(pos_ratings))
            else:
                pos_avgs.append(np.nan)
                pos_min.append(np.nan)
                pos_max.append(np.nan)
        return [total_avg] + pos_avgs + pos_min + pos_max
    def buildStarterMaddenRatings(self, source: pd.DataFrame, sdf: pd.DataFrame, rdf: pd.DataFrame, isNew: bool):
        if 'starterMaddenRatings.csv' in os.listdir(self._dir) and not isNew:
            print('starterMaddenRatings already exists.')
            return
        print('Creating starterMaddenRatings...')
        # total avg, position avg, max/min at position
        cols = [(pos + '_madden_' + suffix) for suffix in ['avg', 'min', 'max'] for pos in self.positions]
        cols = ['total_madden_avg'] + cols
        cols = [(prefix + '_' + col) for prefix in ['home','away'] for col in cols]
        new_df = pd.DataFrame(columns=list(source.columns)+cols)
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), 'StarterMaddenRatings')
            wy, home_abbr, away_abbr = row[['wy', 'home_abbr', 'away_abbr']]
            home_info = self.getInfo(wy, home_abbr, sdf, rdf)
            away_info = self.getInfo(wy, away_abbr, sdf, rdf)
            new_df.loc[len(new_df.index)] = list(row.values) + home_info + away_info
        new_df.fillna(new_df.mean(), inplace=True)
        if not isNew:
            self.saveFrame(new_df, (self._dir + "starterMaddenRatings"))
        return new_df
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
    
# END / StarterMaddenRatings

#########################

# smr = StarterMaddenRatings("./")

# source = pd.read_csv("%s.csv" % "../source/source")
# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")
# rdf = pd.read_csv("%s.csv" % "../../../../maddenRatings/playerRatings")

# smr.buildStarterMaddenRatings(source, sdf, rdf)

# smr = StarterMaddenRatings("./")

# source = pd.read_csv("%s.csv" % "../source/new_source")
# sdf = pd.read_csv("%s.csv" % "../../../../data/starters_23/starters_w2")
# rdf = pd.read_csv("%s.csv" % "../../../../maddenRatings/playerRatings")

# df = smr.buildStarterMaddenRatings(source, sdf, rdf, True)

# df.to_csv("temp.csv", index=False)
