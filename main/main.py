import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../')

from paths import DATA_PATH, POSITION_PATH, STARTERS_PATH, TEAMNAMES_PATH, COACHES_PATH, MADDEN_PATH, NAMES_PATH, SNAP_PATH

# from playerRanks.build import main as prMain
# from gamePredictions.build import main as gpMain, new_main as gpNew
from gamePredictions.build import Build
from gamePredictions.tf_predict import TfPredict
# from fantasyPredictions.build import main as fpMain, new_main as fpNew
from fantasyPredictions.build import Build as FpBuild
from fantasyPredictions.tf_predict import TfPredict as FpTfPredict

class Main:
    def __init__(self, week, year):
        self.all_paths = {
            'dp': DATA_PATH, 'pp': POSITION_PATH, 'sp': STARTERS_PATH,
            'tnp': TEAMNAMES_PATH, 'cp': COACHES_PATH, 'mrp': MADDEN_PATH,
            'sc': SNAP_PATH
        }
        self.week = week
        self.year = year
        self.gp_dir = 'gamePredictions/'
        self.fp_dir = 'fantasyPredictions/'
        return
    # game predictions + points predictions
    def gpPredicitions(self, name):
        if name == 'train':
            b = Build(self.all_paths, self.gp_dir)
            b.main()
        elif name == 'new':
            b = Build(self.all_paths, self.gp_dir)
            b.new_main(self.week, self.year)
            # tfp = TfPredict(self.gp_dir)
            # tfp.build()
            # tfp.buildConsensus()
        else:
            b = Build(self.all_paths, self.gp_dir)
            b.main()
            b.new_main(self.week, self.year)
            # tfp = TfPredict(self.gp_dir)
            # tfp.build()
            # tfp.buildConsensus()
        return
    # fantasy points + rank predictions
    def fpPredicitions(self, name):
        if name == 'train':
            b = FpBuild(self.all_paths, self.fp_dir)
            b.main()
        elif name == 'new':
            b = FpBuild(self.all_paths, self.fp_dir)
            b.new_main(self.week, self.year)
            # tfp = FpTfPredict(self.fp_dir)
            # tfp.build()
        else:
            b = FpBuild(self.all_paths, self.fp_dir)
            b.main()
            b.new_main(self.week, self.year)
            # tfp = FpTfPredict(self.fp_dir)
            # tfp.build()
        return
    # find missing columns
    def findMissingCols_gp(self):
        train = pd.read_csv("%s.csv" % (self.gp_dir + "train"))
        test = pd.read_csv("%s.csv" % (self.gp_dir + "test"))
        # for col in train.columns:
        #     if col not in test.columns:
        #         print(col)
        print(test.isna().any())
        return
    def test(self):
        b = Build(self.all_paths, self.gp_dir)
        # b.test_func()
        b.buildPredTargets()
        return
    # clean gamePredictions
    def cleanGp(self):
        df = pd.read_csv("%s.csv" % (self.gp_dir + "consensus"))
        df = df[[
            'key', 'home_abbr', 'away_abbr', 
            'predictions_points_linear_h_won', 'most_common', 'consensus'
        ]]
        df['best_model'] = df.apply(lambda x: x['home_abbr'] if x['predictions_points_linear_h_won']==1 else x['away_abbr'], axis=1)
        df['most_common'] = df.apply(lambda x: x['home_abbr'] if x['most_common']==1 else x['away_abbr'], axis=1)
        df['confidence'] = df['consensus'] - 25
        df.drop(columns=['predictions_points_linear_h_won', 'consensus'], inplace=True)
        pdf = pd.read_csv("%s.csv" % (self.gp_dir + "predictions"))
        pdf = pdf[['key', 'home_points_linear', 'away_points_linear']]
        pdf.columns = ['key', 'home_points', 'away_points']
        df = df.merge(pdf, on=['key'])
        return df
    # clean fantasyPredictions
    def cleanFp(self):
        str_cols = ['key', 'abbr', 'p_id', 'wy', 'position']
        pos_dfs = {}
        for pos in ['QB', 'RB', 'WR', 'TE']:
            df = pd.read_csv("%s.csv" % (self.fp_dir + "predictions_" + pos))
            point_cols = [col for col in df.columns if 'points' in col]
            df['points'] = df[point_cols].mean(axis=1)
            df = df.drop(columns=point_cols)
            all_stat_cols = [col for col in df.columns if 'points' not in col and col not in str_cols]
            stat_cols = ["_".join(col.split("_")[2:4]) for col in all_stat_cols]
            for stat in stat_cols:
                cols = [col for col in df.columns if stat in col]
                df[stat] = df[cols].mean(axis=1)
            df = df.drop(columns=all_stat_cols)
            df = df.sort_values(by=['points'], ascending=False)
            pos_dfs[pos] = df
        return pos_dfs
    # get name from pid
    def getPlayerName(self, pid: str, df: pd.DataFrame):
        try:
            name = df.loc[df['p_id']==pid, 'name'].values[0]
        except IndexError:
            name = 'UNK'
        return name
    # all player data for UI
    def getAllFpData(self):
        str_cols = ['key', 'abbr', 'p_id', 'wy', 'position']
        keep_cols = ['points_linear', 'week_rank_linear']
        df_list = []
        for pos in ['QB', 'RB', 'WR', 'TE']:
            df = pd.read_csv("%s.csv" % (self.fp_dir + "predictions_" + pos))
            df = df[str_cols+keep_cols+[(pos + '_points_linear')]]
            df.columns = str_cols+['points', 'week_rank', 'indiv_points']
            df['week_rank'] = df['week_rank'].round(0)
            df = df[str_cols+['points', 'indiv_points', 'week_rank']]
            df_list.append(df)
        new_df = pd.concat(df_list)
        pn = pd.read_csv("%s.csv" % (NAMES_PATH + "playerInfo"))
        new_df.insert(1, 'name', new_df['p_id'].apply(lambda x: self.getPlayerName(x, pn)))
        return new_df
    # write csv as json to nfl-app assets directory
    def toJson(self, df: pd.DataFrame, name: str):
        ui_dir = "../nfl-app/src/assets/"
        res = df.to_json(orient='records')
        with open((ui_dir + name + '.json'), 'w') as file:
            file.write(res)
        file.close()
        return
    # create ui data
    def createUiData(self):
        gp_df = self.cleanGp()
        self.toJson(gp_df, 'consensus')
        all_fp_df = self.getAllFpData()
        self.toJson(all_fp_df, 'all_fp_data')
        pos_dfs = self.cleanFp()
        for pos in pos_dfs:
            self.toJson(pos_dfs[pos], (pos + 'Data'))
        return

########################

if __name__ == '__main__':
    m = Main(
        week=13,
        year=2023
    )
    m.gpPredicitions('new')
    # m.fpPredicitions('both')
    # m.findMissingCols_gp()
    # m.test()
    # m.createUiData()
        