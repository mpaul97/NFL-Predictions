import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import time

class Main:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "../data/"
        return
    def get_names(self, text: str):
        soup = BeautifulSoup(text[:text.index('/table')], 'html.parser')
        _dict = {}
        for tag in soup.find_all('a'):
            link = tag.get('href')
            pid = link.split("/")[-1].replace('.htm','')
            name = tag.get_text()
            _dict[name] = pid
        return _dict
    def get_frame(self, text: str):
        df = pd.read_html(text)[0]
        names = self.get_names(text)
        df.columns = [' '.join(col).strip() for col in df.columns.values]
        df.columns = ['name', 'position', 'off_num', 'off_pct', 'def_num', 'def_pct', 'st_num', 'st_pct']
        df.dropna(inplace=True)
        df.insert(0, 'p_id', df['name'].apply(lambda x: names[x]))
        df.drop(columns=['name'], inplace=True)
        for col in [col for col in df.columns if 'pct' in col]:
            df[col] = df[col].apply(lambda x: float(x.replace('%',''))/100)
        return df
    def get_snap_counts(self, key: str, home_abbr: str, away_abbr: str):
        url = "https://www.pro-football-reference.com/boxscores/" + key + ".htm"
        res = requests.get(url)
        home_start = res.text.index('div_home_snap_counts')
        home_df = self.get_frame(res.text[home_start:])
        home_df.insert(0, 'abbr', home_abbr)
        away_start = res.text.index('div_vis_snap_counts')
        away_df = self.get_frame(res.text[away_start:])
        away_df.insert(0, 'abbr', away_abbr)
        return pd.concat([home_df, away_df])
    def build(self):
        cd = pd.read_csv("%s.csv" % (self.data_dir + "gameData"))
        start = cd.loc[cd['wy'].str.contains('2012')].index.values[0]
        cd: pd.DataFrame = cd.loc[cd.index>=start]
        df_list = []
        for index, (key, wy, home_abbr, away_abbr) in enumerate(cd[['key', 'wy', 'home_abbr', 'away_abbr']].values):
            self.printProgressBar(index, cd.shape[0], 'Snap Counts')
            df = self.get_snap_counts(key, home_abbr, away_abbr)
            df.insert(0, 'key', key)
            df.insert(1, 'wy', wy)
            df_list.append(df)
            time.sleep(2)
        new_df = pd.concat(df_list)
        self.save_frame(new_df, (self._dir + "snap_counts"))
        return
    def update(self):
        cd = pd.read_csv("%s.csv" % (self.data_dir + "gameData"))
        start = cd.loc[cd['wy'].str.contains('2012')].index.values[0]
        cd: pd.DataFrame = cd.loc[cd.index>=start]
        cd = cd.reset_index()
        sdf = pd.read_csv("%s.csv" % (self._dir + "snap_counts"))
        keys = list(set(cd['key'].values).difference(set(sdf['key'].values)))
        if len(keys) == 0:
            print('snap_counts up-to-date.')
            return
        print('Updating snap_counts...')
        cd = cd.loc[cd['key'].isin(keys)]
        df_list = []
        for index, (key, wy, home_abbr, away_abbr) in enumerate(cd[['key', 'wy', 'home_abbr', 'away_abbr']].values):
            # self.printProgressBar(index, cd.shape[0], 'Snap Counts')
            df = self.get_snap_counts(key, home_abbr, away_abbr)
            df.insert(0, 'key', key)
            df.insert(1, 'wy', wy)
            df_list.append(df)
            time.sleep(2)
        new_df = pd.concat(df_list)
        new_df = pd.concat([sdf, new_df])
        self.save_frame(new_df, (self._dir + "snap_counts"))
        return
    def save_frame(self, df: pd.DataFrame, name: str):
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

####################

# m = Main("./")
# # m.build()
# m.update()