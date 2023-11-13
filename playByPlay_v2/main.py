import pandas as pd
import numpy as np
import os
import datetime
import random
import tkinter as tk

from guis import LabelSelector, PlayTypes

pd.options.mode.chained_assignment = None

class Main:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.raw_tables_dir = self.data_dir + "raw/"
        self.clean_tables_dir = self.data_dir + "clean/"
        self.labels_dir = self.data_dir + "labels/"
        # external paths
        self.tn_dir = self._dir + "../teamNames/"
        # frames
        self.alt_abbrs = pd.read_csv("%s.csv" % (self.tn_dir + "altAbbrs"))
        self.df: pd.DataFrame = None
        # info
        self.label_cols = ['primary_key', 'key', 'num', 'detail']
        self.play_types = [
            'pass', 'run', 'sack', 'penalty', 'punt',
            'kickoff', 'field_goal', 'extra_point', 'coin_toss',
        ]
        self.valid_quarters = [
            np.nan, '1.0', '2', '4', '4.0', 
            '1', '3', '2.0', 'OT', '3.0',
            1, 2, 3, 4
        ]
        return
    def createPbpColumnInfo(self):
        """
        Find all column structures
        """
        fns = os.listdir(self.raw_tables_dir)
        all_cols = []
        for index, fn in enumerate(fns):
            self.printProgressBar(index, len(fns), 'Collecting columns')
            df = pd.read_csv((self.tables_dir + fn), compression="gzip")
            [all_cols.append(col) for col in df.columns if col not in all_cols]
        with open((self.data_dir + 'pbp_columns.txt'), 'w') as f:
            f.write('\n'.join(all_cols))
        f.close()
        return
    def createAllData(self):
        """
        Concat all tables and write all detail lines to .txt file
        """
        df_list = []
        file = open((self.data_dir + "allDetails.txt"), "w")
        fns = os.listdir(self.clean_tables_dir)
        for index, fn in enumerate(fns):
            self.printProgressBar(index, len(fns), 'Concatenating/creating all data')
            df = pd.read_csv(self.clean_tables_dir + fn)
            df = df.loc[df['quarter'].isin(self.valid_quarters)]
            df.insert(0, 'primary_key', [(fn.replace('.csv','') + '-' + str(i)) for i in range(len(df.index))])
            df.insert(1, 'key', fn.replace('.csv',''))
            df.insert(2, 'num', [i for i in range(len(df.index))])
            df_list.append(df)
            df = df[['detail']]
            df.dropna(inplace=True)
            file.write('\n'.join(df['detail']))
        file.close()
        self.saveFrame(pd.concat(df_list), (self.data_dir + "allTables"))
        return
    def addSelectedLabels(self, df: pd.DataFrame, labelName: str):
        """
        Creates LabelSelector UI to get user input labels
        and adds to dataframe
        Args:
            df (pd.DataFrame): ['primary_key', 'key', 'num', 'detail']
        """
        root = tk.Tk()
        ls = LabelSelector(root, labelName, list(df['detail']))
        root.mainloop()
        df[labelName] = ls.selected_values
        return df
    def addPlayTypeLabels(self, df: pd.DataFrame):
        """
        Creates PlayTypes UI to get user input labels
        and adds to dataframe
        Args:
            df (pd.DataFrame): ['primary_key', 'key', 'num', 'detail']
        """
        root = tk.Tk()
        pt = PlayTypes(root, list(df['detail']))
        root.mainloop()
        for key in pt.selected_values:
            df[key] = pt.selected_values[key]
        return df
    def createLabels(self, labelName: str, game_key: str = ''):
        self.setDf()
        self.df = self.df.dropna(subset=['detail'])
        new_df = pd.DataFrame()
        if (labelName + '.csv') in os.listdir(self.labels_dir):
            new_df = pd.read_csv("%s.csv" % (self.labels_dir + labelName))
            print('Adding to existing...')
        if not new_df.empty: # remove used primary keys
            self.df = self.df.loc[~self.df['primary_key'].isin(new_df['primary_key'])]
        if game_key == '':# random sample of details
            sample = random.sample(list(self.df.index), 100)
            df = self.df.loc[self.df.index.isin(sample)]
            print("Using random sample.")
        else:
            df = self.df.loc[self.df['key']==game_key]
            print(f"Using {game_key}.")
        df = df[self.label_cols]
        df = self.addSelectedLabels(df, labelName)
        self.saveFrame(pd.concat([new_df, df]), (self.labels_dir + labelName))
        return
    def createLabels_playTypes(self):
        self.setDf()
        self.df = self.df.dropna(subset=['detail'])
        o_df = pd.DataFrame()
        if 'play_types.csv' in os.listdir(self.labels_dir):
            o_df = pd.read_csv("%s.csv" % (self.labels_dir + 'play_types'))
            print('Adding to existing play_types...')
        if not o_df.empty: # remove used primary keys
            self.df = self.df.loc[~self.df['primary_key'].isin(o_df['primary_key'])]
        # random sample
        sample = random.sample(list(self.df.index), 100)
        df = self.df.loc[self.df.index.isin(sample)]
        df = df[self.label_cols]
        df = self.addPlayTypeLabels(df)
        self.saveFrame(df, (self.labels_dir + 'play_types'))
        return
    def setDf(self):
        self.df = pd.read_csv("%s.csv" % (self.data_dir + "allTables"))
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
    
# END / Main

###################

# !!! Tables go from 1994 - 2023 week 4 !!!

m = Main("./")

# m.createAllData()

m.createLabels(
    labelName='isExtraPoint'
)
