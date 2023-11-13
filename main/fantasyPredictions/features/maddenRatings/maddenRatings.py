import pandas as pd
import numpy as np
import os
import regex as re

pd.options.mode.chained_assignment = None

class MaddenRatings:
    def __init__(self, _dir):
        self._dir = _dir
        self.positions = ['QB', 'RB', 'WR', 'TE']
        return
    def buildMaddenRatings(self, source: pd.DataFrame, rdf: pd.DataFrame, isNew: bool):
        if 'maddenRatings.csv' in os.listdir(self._dir) and not isNew:
            print('maddenRatings already exists.')
            return
        print("Creating maddenRatings...")
        new_df = pd.DataFrame(columns=list(source.columns)+['madden_rating'])
        for index, row in source.iterrows():
            if not isNew:
                self.printProgressBar(index, len(source.index), 'Madden Ratings')
            pid, wy = row[['p_id', 'wy']]
            year = int(wy.split(" | ")[1])
            try:
                rating = rdf.loc[(rdf['p_id']==pid)&(rdf['year']==year), 'rating'].values[0]
            except:
                rating = np.nan
            new_df.loc[len(new_df.index)] = list(row.values) + [rating]
        new_df.fillna(new_df.mean(), inplace=True)
        if not isNew:
            self.saveFrame(new_df, (self._dir + "maddenRatings"))
        return new_df
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
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
    
# END / MaddenRatings

################################

# mr = MaddenRatings("./")

# source = pd.read_csv("%s.csv" % "../source/source")
# rdf = pd.read_csv("%s.csv" % "../../../../maddenRatings/playerRatings")

# mr.buildMaddenRatings(source, rdf, False)