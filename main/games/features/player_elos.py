import pandas as pd
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
import statsmodels.api as sm
from random import randrange
import seaborn as sns
import pickle
import time
import random

from GUI import GUI

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

pd.options.mode.chained_assignment = None

class PEModels:
    def __init__(self, position: str, df: pd.DataFrame, _dir: str):
        """
        Initializes PEModels class.
        Used to create player_elos models from manual user input.
        @params:
            position   - Required  : lower case position
            df   - Required  : game data (DataFrame)
            _dir   - Required  : directory
        """
        self.position = position
        POSITION_PATH = _dir + '../../../../data/positionData/'
        self.cd: pd.DataFrame = pd.read_csv("%s.csv" % (POSITION_PATH + position.upper() + 'Data'))
        start = self.cd.loc[self.cd['wy']=='1 | 2012'].index.values[0]
        self.cd: pd.DataFrame = self.cd.loc[self.cd.index>=start]
        self.cd: pd.DataFrame = self.cd.reset_index(drop=True)
        # self.cd = pd.concat([self.cd, pd.read_csv("%s.csv" % "qb_mockPositionData_1-2023")])
        # self.cd: pd.DataFrame = self.cd.reset_index(drop=True)
        self.df: pd.DataFrame = df
        # self.df = pd.concat([self.df, pd.read_csv("%s.csv" % "mockGameData_1-2023")])
        # self.df: pd.DataFrame = self.df.reset_index(drop=True)
        self.season_rankings: pd.DataFrame = None
        self.pe_custom_cols = {
            'qb': [
                'total_yards', 'total_touchdowns', 'pass_attempts_percentage',
                'points_for', 'points_against', 'won', 
                'pass_yards_rank', 'allowed_pass_yards_rank'
            ],
        }
        self.pe_train_cols = {
            'qb': [
                'total_yards', 'total_touchdowns', 'interceptions_thrown',
                'quarterback_rating', 'adjusted_yards_per_attempt', 'completion_percentage', 
                'pass_attempts_percentage', 'points_for', 'points_against',
                'won', 'pass_yards_rank', 'allowed_pass_yards_rank'
                
            ],
            'rb': [
                'yards_from_scrimmage', 'scrimmage_yards_per_touch', 
                'receiving_touchdowns', 'total_touches', 'volume_percentage',
                'rush_touchdowns', 'longest_rush'
            ],
            'wr': [
                'times_pass_target', 'receptions', 'yards_from_scrimmage',
                'receiving_touchdowns', 'receiving_yards_per_reception', 'catch_percentage',
                'longest_reception'
            ]
        }
        self.ignored_cols = {
            'qb': 'pass_attempts_percentage'
        }
        self.train_funcs = {
            'qb': self.get_train_qb, 'rb': self.get_train_rb, 'wr': self.get_train_wr
        }
        # directories
        self._dir: str = _dir
        self.main_dir: str = self._dir + 'player_elos/'
        self.mc_dir(self.main_dir)
        # train
        self.train_dir: str = self.main_dir + 'per_game_train/'
        self.train_dir: str = self.mc_dir(self.train_dir, position)
        # # target
        self.target_dir: str = self.main_dir + 'per_game_targets/'
        self.target_dir: str = self.mc_dir(self.target_dir, position)
        # models
        self.models_dir: str = self.main_dir + 'models/'
        self.mc_dir(self.models_dir)
        # data
        self.data_dir: str = self.main_dir + 'data/'
        self.data_dir: str = self.mc_dir(self.data_dir, position)
        # model stuff
        self.model: RandomForestClassifier = None
        self.best_cols = []
        # grade stuff
        self.game_grades: pd.DataFrame = None # per game grades
        return
    def mc_dir(self, _dir: str, position: str = None):
        """
        Check if directory exists. Make if not.
        Args:
            _dir (str): directory
            parent_dir (str): parent directory
        """
        try:
            os.mkdir(_dir)
        except FileExistsError:
            pass
        if position:
            p_dir: str = _dir + position + '/'
            try:
                os.mkdir(p_dir)
            except FileExistsError:
                pass
            return p_dir
        return
    def zero_division(self, num, dem):
        try:
            return num/dem
        except ZeroDivisionError:
            return 0
    def set_season_rankings(self):
        self.season_rankings = pd.read_csv("%s.csv" % (self._dir + "season_rankings"))
        return
    def get_train_qb(self, cd: pd.DataFrame):
        """
        Returns train data for QBs.
        @params:
            cd   - Required  : position player data (only pg_train cols and/or only 1 wy data for individuals)
        """
        def func(row: pd.Series):
            is_home = row['abbr'] == row['home_abbr']
            _dict = {}
            _dict['total_yards'] = row['passing_yards'] + row['rush_yards']
            _dict['total_touchdowns'] = row['passing_touchdowns'] + row['rush_touchdowns']
            _dict['pass_attempts_percentage'] = self.zero_division(row['attempted_passes'], row['home_pass_attempts']) if is_home else self.zero_division(row['attempted_passes'], row['away_pass_attempts'])
            _dict['points_for'] = row['home_points'] if is_home else row['away_points']
            _dict['points_against'] = row['away_points'] if is_home else row['home_points']
            _dict['won'] = 1 if row['abbr'] == row['winning_abbr'] else 0
            _dict['pass_yards_rank'] = row['home_pass_yards_rank'] if is_home else row['away_pass_yards_rank']
            _dict['allowed_pass_yards_rank'] = row['home_allowed_pass_yards_rank'] if is_home else row['away_allowed_pass_yards_rank']
            return _dict
        df: pd.DataFrame = self.df.copy()[[
            'key', 'wy', 'home_abbr', 'away_abbr', 
            'winning_abbr', 'home_points', 'away_points', 
            'home_pass_attempts', 'away_pass_attempts'
        ]]
        cd.columns = ['key' if col == 'game_key' else col for col in cd.columns]
        cd: pd.DataFrame = cd.merge(df, on=['key', 'wy'])
        sr: pd.DataFrame = self.season_rankings.copy()[[
            'key', 'wy', 'home_pass_yards_rank',
            'away_pass_yards_rank', 'home_allowed_pass_yards_rank', 'away_allowed_pass_yards_rank'
        ]]
        cd: pd.DataFrame = cd.merge(sr, on=['key', 'wy'])
        cd[self.pe_custom_cols[self.position]] = cd.apply(lambda x: func(x), result_type='expand', axis=1)
        cd = cd[['p_id', 'wy', 'key', 'abbr'] + self.pe_train_cols[self.position]]
        return cd
    def get_train_rb(self, cd: pd.DataFrame):
        """
        Returns train data for RBs.
        @params:
            cd   - Required  : position player data (only pg_train cols and/or only 1 wy data for individuals)
        """
        cd['total_touchdowns'] = cd['rush_touchdowns'] + cd['receiving_touchdowns']
        cd.drop(columns=[
            'rush_touchdowns', 'receiving_touchdowns'
        ], inplace=True)
        # cd = cd.loc[cd['volume_percentage']>0.2] # filter inactive players
        return cd
    def get_train_wr(self, cd: pd.DataFrame):
        """
        Returns train data for WRs.
        @params:
            cd   - Required  : position player data (only pg_train cols and/or only 1 wy data for individuals)
        """
        # cd = cd.loc[cd['volume_percentage']>0.1] # filter inactive players
        return cd
    def write_train_data(self):
        """
        Saves train data to per_game_train folder.
        @params:
            wy   - Required  : determines if train data is save for ALL
        """
        self.set_season_rankings()
        cd = self.cd.copy()
        cd.reset_index(drop=True, inplace=True)
        cd: pd.DataFrame = self.train_funcs[self.position](cd)
        cd: pd.DataFrame = cd[['p_id', 'wy', 'key', 'abbr']+self.pe_train_cols[self.position]]
        cd = cd.round(2)
        self.save_frame(cd, (self.train_dir + self.position + "_train_all"))
        print(self.position + "_train_all.csv written to " + self.train_dir)
        return
    def build_target_data(self, wy: str):
        fn_wy = wy.replace(' | ','-')
        target_fn = self.target_dir + self.position + "_target_" + fn_wy
        df: pd.DataFrame = self.get_train_data(wy)
        # remove rows where player didn't play
        df: pd.DataFrame = df.loc[df[self.ignored_cols[self.position]]>0.1]
        dp = GUI(df)
        grades = dp.radio_values
        new_df = df[['p_id', 'wy', 'key']]
        new_df['grade'] = grades
        new_df.to_csv("%s.csv" % target_fn, index=False)
        print(target_fn + ' saved.')
        return
    def show_heat_map(self):
        target: pd.DataFrame = pd.concat([pd.read_csv(self.target_dir + fn) for fn in os.listdir(self.target_dir) if 'all' not in fn])
        data: pd.DataFrame = target.merge(self.get_train_data(), on=['p_id', 'wy', 'key'])
        corrmat = data.corr()
        k = 20
        cols = corrmat.nlargest(k, 'grade')['grade'].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=0.75)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        return
    def get_corr_cols(self, data: pd.DataFrame, k):
        corrmat = data.corr()
        cols = list(corrmat.nlargest(k, 'grade')['grade'].index)
        cols.remove('grade')
        return cols
    def save_model(self):
        target: pd.DataFrame = pd.concat([pd.read_csv(self.target_dir + fn) for fn in os.listdir(self.target_dir) if 'all' not in fn])
        data: pd.DataFrame = target.merge(self.get_train_data(), on=['p_id', 'wy', 'key'])
        best_cols = self.get_corr_cols(data, 20)
        with open(self.models_dir + self.position +  '_best_cols.txt', 'w') as f:
            f.write(','.join(best_cols))
        f.close()
        X = data[best_cols]
        y = data['grade']
        all_models = []
        for _ in range(50):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # model = LinearRegression(n_jobs=-1)
            model = RandomForestRegressor(n_jobs=-1)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            all_models.append((model, acc))
        all_models.sort(key=lambda x: x[1], reverse=True)
        best_model: LinearRegression = all_models[0][0]
        print(f"Best model accuracy: {all_models[0][1]}")
        pickle.dump(best_model, open((self.models_dir + self.position + '_model.sav'), 'wb'))
        return
    def set_model(self):
        self.model = pickle.load(open((self.models_dir + self.position + '_model.sav'), 'rb'))
        return
    def set_best_cols(self):
        f = open((self.models_dir + self.position + '_best_cols.txt'), 'r')
        self.best_cols = f.read().split(",")
        return
    def test_model(self, wy: str):
        """
        Prints top-2 best columns along with their predicted grades.
        @params:
            wy   - Required  : determines week being tested.
        """
        self.set_model()
        self.set_best_cols()
        cd = self.get_train_data().loc[self.cd['wy']==wy]
        X = cd[self.best_cols]
        preds = self.model.predict(X)
        cd = cd[['p_id'] + self.best_cols[:2]]
        cd['grade'] = preds
        cd.sort_values(by=['grade'], ascending=False, inplace=True)
        print(cd)
        return
    def predict_grades(self):
        """
        Loads saved model, predicts all per game grades, and writes to data_dir.
        """
        self.set_model()
        self.set_best_cols()
        df = pd.read_csv("%s.csv" % (self.train_dir + self.position + "_train_all"))
        preds = self.model.predict(df[self.best_cols])
        df = df[['p_id', 'wy', 'key', 'abbr']]
        df['grade'] = preds
        df = df.round(2)
        df.to_csv("%s.csv" % (self.data_dir + self.position + "_game_grades"), index=False)
        print(f"{self.position}_game_grades written to data_dir.")
        return
    def build_mock_data(self, wy: str):
        """
        Copys game and position data from 5 | 2022. Converts key and wy for mock testing.
        @params:
            wy   - Required  : specifies new wy for mock data (str)
        """
        old_wy = '5 | 2022'
        # game data
        df = self.df
        df = df.loc[df['wy']==old_wy]
        df['key'] = [('MOCK_' + str(i)) for i in range(len(df.index))]
        df['wy'] = wy
        self.save_frame(df, (self._dir + "mock_game_data_" + wy.replace(" | ","-")))
        # position data
        cd = self.cd
        cd = cd.loc[cd['wy']==old_wy]
        keys = []
        for index, row in cd.iterrows():
            abbr = row['abbr']
            new_key = df.loc[(df['home_abbr']==abbr)|(df['away_abbr']==abbr), 'key'].values[0]
            keys.append(new_key)
        cd['game_key'] = keys
        cd['wy'] = wy
        self.save_frame(cd, (self._dir + self.position + "_mock_position_data_" + wy.replace(" | ","-")))
        return
    def get_train_data(self, wy: str = None):
        """
        Get per_game_train data ALL or for wy, if specified
        """
        df = pd.read_csv("%s.csv" % (self.train_dir + self.position + '_train_all'))
        if wy:
            df = df.loc[df['wy']==wy]
            df.reset_index(drop=True, inplace=True)
        return df
    def get_position_game_data(self, wy: str):
        """
        Returns position game data to per_game_train.
        @params:
            wy   - Required  : determines if train data is save for singular wy
        """
        cd = self.cd.loc[self.cd['wy']==wy]
        cd.reset_index(drop=True, inplace=True)
        cd: pd.DataFrame = cd[['p_id', 'wy', 'key', 'abbr']+self.pe_train_cols[self.position]]
        cd = self.train_funcs[self.position](cd)
        cd = cd.round(2)
        return cd
    def update(self):
        """
        Updates game_grades with for unseen wy in gameData/positionData.
        """
        self.set_game_grades()
        self.set_best_cols()
        self.set_model()
        last_wy_gg = self.game_grades['wy'].values[-1]
        last_wy_gd = self.df['wy'].values[-1]
        if last_wy_gg != last_wy_gd:
            df = self.get_position_game_data(last_wy_gd)
            preds = self.model.predict(df[self.best_cols])
            df = df[['p_id', 'wy', 'key', 'abbr']]
            df['grade'] = preds
            df = df.round(2)
            self.game_grades = pd.concat([self.game_grades, df])
            self.save_frame(self.game_grades, (self.data_dir + self.position + "_game_grades"))
            print(f"{self.position}_game_grades updated for {last_wy_gd} and written to data_dir.")
        else:
            print(f"{self.position}_game_grades already up-to-date.")
        return
    def test_game_grades(self, pid: str):
        """
        Display grades for given player.
        Args:
            pid (str): player id
        """
        self.set_game_grades()
        df = self.game_grades.loc[self.game_grades['p_id']==pid]
        print(df)
        return
    def set_game_grades(self):
        self.game_grades = pd.read_csv("%s.csv" % (self.data_dir + self.position + "_game_grades"))
        return
    # write dataframe
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    
# / END Models

#########################

# df = pd.read_csv("%s.csv" % "../../../data/gameData")

# pem = PEModels('qb', df, './data/')

# pem.write_train_data()

# pem.build_target_data('16 | 2023')

# pem.show_heat_map()

# pem.save_model()

# pem.test_model('18 | 2023')

# pem.predict_grades()

# pem.test_game_grades('LoveJo03')

# pem.build_mock_data('1 | 2024')

# pem.update()

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

class Elo:
    def __init__(self, pid: str, elo: float):
        self.pid = pid
        self.elo = elo
        return

class PlayerGrades:
    def __init__(self, position, _dir):
        self.position = position
        self.game_grades: pd.DataFrame = None
        self.elos: pd.DataFrame = None
        self._dir = _dir
        self.main_dir: str = self._dir + 'player_elos/'
        self.data_dir: str = self.main_dir + 'data/' + position + '/'
        self.grade_funcs = {
            'qb': self.get_grade_qb, 'rb': self.get_grade_rb, 'wr': self.get_grade_wr
        }
        self.start: float = 0
        self.end: float = 0
        return
    def build_elos(self):
        self.start = time.time()
        """
        Creates and writes player elos to data_dir.
        """
        if (self.position + '_elos.csv') in os.listdir(self.data_dir):
            print((self.position + '_elos.csv') + ' already built. Proceeding to build.')
            return
        self.set_game_grades()
        df: pd.DataFrame = self.game_grades
        # df = df.loc[df['wy'].str.contains('2022')]
        # df.reset_index(drop=True, inplace=True)
        first_wy = df['wy'].values[0]
        all_pids = list(set(df['p_id']))
        all_pids.sort()
        end_pids = all_pids.copy()
        new_df = pd.DataFrame(columns=['p_id', 'wy', 'elo'])
        # init
        for pid in all_pids:
            new_df.loc[len(new_df.index)] = [pid, first_wy, 100]
        total_mean = np.mean(df['grade'].values)
        for index, vals in enumerate(df[['p_id', 'wy', 'grade']].values):
            self.print_progress_bar(index, len(df.index), 'Building elos')
            pid, wy, grade = vals
            year = int(wy.split(" | ")[1])
            mean = self.get_mean_grade(df, pid, year)
            # grade = (grade - 1) if grade < mean else grade
            # grade = (((grade - mean)*1.5) + (mean - total_mean))
            grade = self.grade_funcs[self.position](grade, mean, total_mean)
            curr_elo = new_df.loc[new_df['p_id']==pid, 'elo'].values[-1]
            try:# add elo to next week - elo going into the week
                next_wy = df.iloc[df.loc[df['wy']==wy].index.values[-1]+1]['wy']
                year = int(wy.split(" | ")[1])
                next_year = int(next_wy.split(" | ")[1])
                new_elo = curr_elo + grade
                if year == next_year: # same year
                    new_df.loc[len(new_df.index)] = [pid, next_wy, new_elo]
                else: # new year
                    new_df.loc[len(new_df.index)] = [pid, next_wy, self.get_end_elo(new_elo)]
                    try:
                        end_pids.remove(pid)
                    except ValueError:
                        pass
                    if wy != df.iloc[index+1]['wy']: # add missing abbrs - didnt play last week of prev year
                        for pid1 in end_pids:
                            last_elo = new_df.loc[new_df['p_id']==pid1, 'elo'].values[-1]
                            new_df.loc[(len(new_df.index))] = [pid1, next_wy, self.get_end_elo(last_elo)]
                            end_pids = all_pids.copy()
            except IndexError: # add elo end (END OF DATA) - getElo and then get_end_elo for END(year)
                end_year = int(wy.split(" | ")[1])
                new_wy = '1 | ' + str(end_year+1)
                new_df.loc[len(new_df.index)] = [pid, new_wy, self.get_end_elo(curr_elo + grade)]
                try:
                    end_pids.remove(pid)
                except ValueError:
                    pass
            # # remove pids, no games in future
            # future_pids = df.loc[df.index>index, 'p_id'].values
            # if pid not in future_pids:
            #     try:
            #         all_pids.remove(pid)
            #     except ValueError:
            #         pass
        for pid1 in end_pids: # add pids not present in last week of season
            last_elo = new_df.loc[new_df['p_id']==pid1, 'elo'].values[-1]
            new_df.loc[(len(new_df.index))] = [pid1, new_wy, self.get_end_elo(last_elo)]
        self.save_frame(new_df, (self.data_dir + self.position + "_elos"))
        self.set_elos()
        self.end = time.time()
        elapsed = self.end - self.start
        print(f"Time elapsed: {elapsed}")
        return
    def build_elosV2(self):
        """
        Creates and writes player elos to data_dir.
        """
        # if (self.position + '_elos.csv') in os.listdir(self.data_dir):
        #     print((self.position + '_elos.csv') + ' already built. Proceeding to build.')
        #     return
        self.set_game_grades()
        df: pd.DataFrame = self.game_grades
        # df = df.loc[df['p_id']=='McCaCh01']
        df = df.loc[df['wy'].str.contains('2022')]
        df.reset_index(drop=True, inplace=True)
        all_pids = list(set(df['p_id']))
        all_pids.sort()
        total_mean = np.mean(df['grade'].values)
        new_df = pd.DataFrame(columns=['p_id', 'wy', 'elo'])
        for pid in all_pids:
            data: pd.DataFrame = df.loc[df['p_id']==pid]
            data.reset_index(drop=True, inplace=True)
            elo = 100
            for index, row in data.iterrows():
                wy = row['wy']
                year = int(wy.split(" | ")[1])
                grade = row['grade']
                mean = self.get_mean_grade(data, pid, year)
                grade = self.grade_funcs[self.position](grade, mean, total_mean)
                elo += grade
                if index != data.index[-1]:
                    next_wy = data.iloc[index+1]['wy']
                    next_year = int(next_wy.split(" | ")[1])
                    if year != next_year:
                        elo = self.get_end_elo(elo)
                    new_df.loc[len(new_df.index)] = [pid, next_wy, elo]
            # add first row - elo
            first_wy = data['wy'].values[0]
            first_row = [{'p_id': pid, 'wy': first_wy, 'elo': 100}]
            new_df = pd.concat([pd.DataFrame(first_row), new_df])
            # add last row - elo
            last_wy = data['wy'].values[-1]
            last_year = int(last_wy.split(" | ")[1])
            if last_year == 2022:
                new_wy = '1 | ' + str(last_year+1)
                new_df.loc[len(new_df.index)] = [pid, new_wy, self.get_end_elo(elo)]
        self.save_frame(new_df, (self.data_dir + self.position + '_elos_v2'))
        return
    def get_mean_grade(self, df: pd.DataFrame, pid: str, year: int):
        data = df.loc[(df['p_id']==pid)&(df['wy'].str.contains(str(year))), 'grade'].values
        return np.mean(data) if len(data) != 0 else np.mean(df['grade'].values)
    def get_grade_qb(self, grade: float, mean: float, total_mean: float):
        grade = (grade - 1) if grade < mean else grade
        grade = (((grade - mean)*1.5) + (mean - total_mean))
        return grade
    def get_grade_rb(self, grade: float, mean: float, total_mean: float):
        return grade
    def get_grade_wr(self, grade: float, mean: float, total_mean: float):
        return grade
    def graph_elos(self, pids: list, year: int):
        """
        Plots player elos.
        @params:
            pids   - Required  : pids to be plotted (list)
            year   - Required  : year for pids to be plotted (int)
        """
        self.set_elos()
        df = self.elos.loc[(self.elos['wy'].str.contains(str(year)))&(self.elos['p_id'].isin(pids))]
        for pid in pids:
            plt.plot(df.loc[df['p_id']==pid, 'elo'].values)
        plt.legend(pids)
        plt.show()
        return
    def test_elo_function(self):
        """
        Test/find good function for elo calculation.
        """
        n = 5
        mean = 3.3694273127753305
        a_grades = [4.5, 3.3, 4.3, 4.4, 2.6, 4.7]
        b_grades = [2.5, 3.1, 2.8, 2.4, 3.2, 3.3]
        elos = [100]
        info = {
            'a': {
                'grades': [random.uniform(3.5, 4.1) for _ in range(n)],
                'elos': [100]
            },
            'b': {
                'grades': [random.uniform(2.3, 4) for _ in range(n)],
                'elos': [100]
            }
        }
        for key in info:
            grades = info[key]['grades']
            elos = info[key]['elos']
            for grade in grades:
                print(f"Normal grade: {grade}")
                # grade = (grade - 3) if grade < mean else grade
                grade = (grade - mean)*5
                print(f"New grade: {grade}")
                info[key]['elos'].append(elos[-1] + grade)
        #     plt.plot(info[key]['elos'])
        # plt.show()
        return
    def check_best_elos(self, wy: str):
        self.set_elos()
        df = self.elos.loc[self.elos['wy']==wy, ['p_id', 'elo']]
        df.sort_values(by=['elo'], ascending=False, inplace=True)
        for index, row in df.iterrows():
            pid, elo = row[['p_id', 'elo']]
            print(f"p_id: {pid}, elo: {elo}")
        return
    def build(self, source: pd.DataFrame, sdf: pd.DataFrame, isNew: bool):
        """
        Converts elos to home vs. away data. Using player with highest elo for given team.
        Returns and prints message if (position + "_playerGrades.csv) already exists.
        @params:
            source   - Required  : source info (DataFrame)
            sdf      - Required  : all starters (DataFrame)
            isNew    - Required  : determines all train or new week (bool)
        """
        if (self.position + "_player_grades.csv") in os.listdir(self._dir) and not isNew:
            print(self.position + "_player_grades.csv already built. Using exisiting.")
            return
        self.set_elos()
        new_df = pd.DataFrame(columns=list(source.columns)+[('home_' + self.position + '_elo'), ('away_' + self.position + '_elo')])
        for index, row in source.iterrows():
            key = row['key']
            wy = row['wy']
            home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
            try:
                home_starters = (sdf.loc[(sdf['key']==key)&(sdf['abbr']==home_abbr), 'starters'].values[0]).split("|")
                away_starters = (sdf.loc[(sdf['key']==key)&(sdf['abbr']==away_abbr), 'starters'].values[0]).split("|")
            except IndexError:
                home_starters = []
                away_starters = []
            if len(home_starters) != 0:
                try:
                    home_starters = self.get_starter_elos(home_starters, wy)
                    home_elo = list(home_starters.values())[0]
                except IndexError:
                    home_elo = 100
            if len(away_starters) != 0:
                try:
                    away_starters = self.get_starter_elos(away_starters, wy)
                    away_elo = list(away_starters.values())[0]
                except IndexError:
                    away_elo = 100
            new_df.loc[len(new_df.index)] = list(row.values) + [home_elo, away_elo]
        if not isNew:
            self.save_frame(new_df, (self._dir + self.position + "_player_grades"))
        return new_df
    def get_starter_elos(self, starters: list, wy: str):
        starters = self.get_position_starters(starters)
        df = self.elos
        info = {}
        for s in starters:
            try:
                elo = df.loc[(df['p_id']==s)&(df['wy']==wy), 'elo'].values[0]
            except IndexError: # last elo if not present for wy
                elo = df.loc[df['p_id']==s, 'elo'].values[-1]
            info[s] = elo
        info = dict(sorted(info.items(), key=lambda item: item[1], reverse=True))
        return info
    def get_position_starters(self, starters: list):
        return [s.split(":")[0] for s in starters if s.split(":")[1] == self.position.upper()]
    def update(self, isNewYear: bool):
        """
        Updates elos for new week.
        @params:
            isNewYear   - Required  : determines if week or year will be incremented (bool)
        """
        self.set_game_grades()
        self.set_elos()
        last_wy_gg = self.game_grades['wy'].values[-1]
        new_wy_gg = self.get_new_wy(last_wy_gg, isNewYear)
        last_wy_e = self.elos['wy'].values[-1]
        new_wy_e = self.get_new_wy(last_wy_e, isNewYear)
        if new_wy_gg == new_wy_e:
            df: pd.DataFrame = self.game_grades.loc[self.game_grades['wy']==last_wy_gg]
            total_mean = np.mean(self.game_grades['grade'].values)
            new_df = pd.DataFrame(columns=self.elos.columns)
            for index, row in df.iterrows():
                pid = row['p_id']
                grade = row['grade']
                wy = row['wy']
                year = int(wy.split(" | ")[1])
                mean = self.get_mean_grade(self.game_grades, pid, year)
                grade = (grade - 1) if grade < mean else grade
                grade = (((grade - mean)*1.5) + (mean - total_mean))
                try:
                    prev_elo = self.elos.loc[self.elos['p_id']==pid, 'elo'].values[-1]
                except IndexError: # unseen p_id (new player)
                    prev_elo = 100
                elo = (prev_elo + grade) if not isNewYear else self.get_end_elo(prev_elo + grade)
                new_df.loc[len(new_df.index)] = [pid, new_wy_e, elo]
            self.elos = pd.concat([self.elos, new_df])
            self.save_frame(self.elos, (self.data_dir + self.position + "_elos"))
        else:
            print(f"{self.position}_elos already up-to-date.")
        return
    def get_new_wy(self, wy: str, isNewYear: str):
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        return (str(week+1) + " | " + str(year)) if not isNewYear else ("1 | " + str(year+1))
    def get_end_elo(self, elo):
        # elo = ((elo * 0.75) + (0.25 * 20))
        return 100 + ((elo - 100)/10) if self.position == 'qb' else 100
        # return 100
    def set_game_grades(self):
        self.game_grades = pd.read_csv("%s.csv" % (self.data_dir + self.position + "_game_grades"))
        return
    def set_elos(self):
        self.elos = pd.read_csv("%s.csv" % (self.data_dir + self.position + "_elos"))
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
    
############################

pg = PlayerGrades('qb', './data/')

# pg.build_elos()

# pg.check_best_elos('14 | 2023')

# pg.graph_elos(['JackLa00', 'MahoPa00', 'PurdBr00', 'LoveJo03'], 2023)

# source = pd.read_csv("%s.csv" % "../source/source")
# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")

# pg.createBoth(source, sdf, False)

# pg.update(isNewYear=False)

# source = pd.read_csv("%s.csv" % "../source/new_source")
# sdf = pd.read_csv("%s.csv" % "../../../../data/starters_23/starters_w2")

# pg.createBoth(source, sdf, True)

# arr = np.arange(58, 123, 5)

# for val in arr:
#     new_val = 100 + ((val - 100)/10)
#     print(f"val, new_val: {val}, {new_val}")