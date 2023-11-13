import pandas as pd
import numpy as np
import os

class Train:
    def __init__(self, league_type: str, _dir: str):
        self.league_type: str = league_type
        self._dir = _dir
        self.data_dir = self._dir + 'data/'
        self.sims_dir = self.data_dir + "sims/"
        self.features_dir = self.data_dir + "features/"
        self.json_dir = self._dir + "../NFL_fantasyProjections/data/"
        self.sim_df: pd.DataFrame = None
        self.source: pd.DataFrame = None
        self._types = ['std', 'ppr', 'half']
        self.frames = { t: pd.read_csv("%s.csv" % (self.json_dir + "json_frame_" + t)) for t in self._types }
        self.source_cols = ['sim_num', 'round', 'drafter']
        return
    def buildSource(self):
        self.setSimDf()
        df = self.sim_df
        new_df = df[self.source_cols]
        self.saveFrame(new_df, (self.data_dir + 'source_' + self.league_type))
        return
    def buildTarget(self):
        self.setSource()
        self.setSimDf()
        df = self.sim_df
        df = df[self.source_cols+['name']]
        new_df = self.source.merge(df, on=self.source_cols)
        cd: pd.DataFrame = self.frames[self.league_type]
        positions = []
        for name in new_df['name'].values:
            pos = cd.loc[cd['name']==name, 'position'].values[0]
            positions.append(pos)
        new_df['position'] = positions
        new_df.drop(columns=['name'], inplace=True)
        self.saveFrame(new_df, (self.data_dir + 'target_' + self.league_type))
        return
    def buildTrain(self):
        self.setSource()
        self.setSimDf()
        df = self.sim_df
        return
    def setSimDf(self):
        df_list = []
        for fn in os.listdir(self.sims_dir):
            if 'sim' in fn and self.league_type in fn:
                df = pd.read_csv(self.sims_dir + fn)
                sim_num = int(fn.replace('sim_','').replace((self.league_type + '_'), '').replace('.csv',''))
                df.insert(0, 'sim_num', sim_num)
                df_list.append(df)
        self.sim_df = pd.concat(df_list)
        return
    def setSource(self):
        self.source = pd.read_csv("%s.csv" % (self.data_dir + "source_" + self.league_type))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    
# END / Train

##########################

t = Train(league_type="std", _dir="./")

# t.buildSource()

# t.buildTarget()

t.buildTrain()

# features
# round, position, current team
# ranks, value(position size), projections

# target
# position + rank