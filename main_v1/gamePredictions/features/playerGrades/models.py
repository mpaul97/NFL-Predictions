import pandas as pd
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
import statsmodels.api as sm
from random import randrange
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

pd.options.mode.chained_assignment = None

class DataFrameDisplay:
    def __init__(self, df):
        self.b_color = '#393e41'
        self.o_color = '#e94f37'
        self.w_color = '#f6f7eb'
        self.df = df
        self.row_index = 0
        self.radio_values = []
        self.root = tk.Tk()
        self.root.configure(background=self.b_color)
        self.pid = tk.Label(self.root)
        self.pid.pack(pady=30)
        # Create a Tkinter label to display the row
        self.label = tk.Label(self.root)
        self.label.pack(padx=50, pady=(0, 30))
        # Create radio buttons
        self.radio_var = tk.IntVar()
        for i in range(1, 6):
            radio_button = tk.Radiobutton(
                self.root, 
                text=str(i), 
                variable=self.radio_var, value=i,
                foreground=self.o_color,
                background=self.w_color,
                activeforeground=self.b_color,
                activebackground=self.o_color,
                selectcolor=self.o_color,
                width=5,
                height=5,
                indicatoron=0,
                font=('Helvetica', 10)
            )
            radio_button.pack(anchor=tk.W, side=tk.LEFT)
        # Create a submit button
        self.submit_button = tk.Button(
            self.root, 
            text="Submit", 
            command=self.submit,
            width=10,
            height=3,
            foreground=self.o_color,
            font=('Helvetica', 16)
        )
        self.submit_button.pack(anchor=tk.W, side=tk.RIGHT)
        # Display the first row
        self.display_row()
        self.root.mainloop()
        return
    def display_row(self):
        row: pd.Series = self.df.iloc[self.row_index]
        pid = row['p_id']
        row = row[1:]
        row_str = '\n'.join([f"{col}: {value}" for col, value in row.items()])
        self.pid.configure(text=pid, foreground=self.o_color, background=self.b_color, font=('Helvetica', 25))
        self.label.configure(text=row_str, background=self.b_color, foreground=self.w_color, font=('Helvetica', 20))
        return
    def submit(self):
        selected_value = self.radio_var.get()
        self.radio_values.append(selected_value)
        self.row_index += 1
        if self.row_index >= len(self.df):
            self.root.destroy()
        else:
            self.display_row()
        return
            
# / END DataFrameDisplay
            
class PgModels:
    def __init__(self, position, df: pd.DataFrame, _dir):
        """
        Initializes Models class.
        @params:
            position   - Required  : lower case position
            df   - Required  : game data (DataFrame)
            _dir   - Required  : directory
        """
        self.position = position
        POSITION_PATH = _dir + '../../../../data/positionData/'
        self.cd: pd.DataFrame = pd.read_csv("%s.csv" % (POSITION_PATH + position.upper() + 'Data'))
        start = self.cd.loc[self.cd['wy']=='1 | 1994'].index.values[0]
        self.cd: pd.DataFrame = self.cd.loc[self.cd.index>=start]
        self.cd: pd.DataFrame = self.cd.reset_index(drop=True)
        # self.cd = pd.concat([self.cd, pd.read_csv("%s.csv" % "qb_mockPositionData_1-2023")])
        # self.cd: pd.DataFrame = self.cd.reset_index(drop=True)
        self.df = df
        # self.df = pd.concat([self.df, pd.read_csv("%s.csv" % "mockGameData_1-2023")])
        # self.df: pd.DataFrame = self.df.reset_index(drop=True)
        self.pg_train_cols = {
            'qb': [
                'passing_yards', 'passing_touchdowns', 'interceptions_thrown',
                'quarterback_rating', 'rush_yards', 'rush_touchdowns',
                'completion_percentage', 'adjusted_yards_per_attempt', 'attempted_passes'
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
        self.train_funcs = {
            'qb': self.getTrain_qb, 'rb': self.getTrain_rb, 'wr': self.getTrain_wr
        }
        # directories
        self._dir = _dir
        self.train_dir = _dir + 'perGameTrain/' + position + '/'
        self.target_dir = _dir + 'perGameTargets/' + position + '/'
        self.models_dir = _dir + 'models/'
        self.data_dir = _dir + 'data/' + position + '/'
        # model stuff
        self.model: RandomForestClassifier = None
        self.best_cols = []
        # grade stuff
        self.game_grades: pd.DataFrame = None # per game grades
        return
    def zeroDivision(self, num, dem):
        try:
            return num/dem
        except ZeroDivisionError:
            return 0
    def getTrain_qb(self, cd: pd.DataFrame):
        """
        Returns train data for QBs.
        @params:
            cd   - Required  : position player data (only pg_train cols and/or only 1 wy data for individuals)
        """
        pointsFor, pointsAgainst, wins, passAttemptsPerc = [], [], [], []
        for index, row in cd.iterrows():
            if len(cd.index) > 100:
                self.printProgressBar(index, len(cd.index), 'Getting train-QB')
            key = row['game_key']
            abbr = row['abbr']
            game = self.df.loc[self.df['key']==key]
            home_abbr = game['home_abbr'].values[0]
            away_abbr = game['away_abbr'].values[0]
            winning_abbr = game['winning_abbr'].values[0]
            home_points = game['home_points'].values[0]
            away_points = game['away_points'].values[0]
            home_pass_attempts = game['home_pass_attempts'].values[0]
            away_pass_attempts = game['away_pass_attempts'].values[0]
            wins.append(1) if winning_abbr == abbr else wins.append(0)
            attempted_passes = row['attempted_passes']
            if abbr == home_abbr:
                pointsFor.append(home_points)
                pointsAgainst.append(away_points)
                if attempted_passes == 0:
                    passAttemptsPerc.append(0)
                else:
                    passAttemptsPerc.append(self.zeroDivision(attempted_passes, home_pass_attempts))
            else:
                pointsFor.append(away_points)
                pointsAgainst.append(home_points)
                if attempted_passes == 0:
                    passAttemptsPerc.append(0)
                else:
                    passAttemptsPerc.append(self.zeroDivision(attempted_passes, away_pass_attempts))
        # / END for
        cd['points_for'] = pointsFor
        cd['points_against'] = pointsAgainst
        cd['won'] = wins
        cd['pass_attempts_perc'] = passAttemptsPerc
        cd['total_yards'] = cd['passing_yards'] + cd['rush_yards']
        cd['total_touchdowns'] = cd['passing_touchdowns'] + cd['rush_touchdowns']
        cd.drop(
            columns=[
                'passing_yards', 'rush_yards', 'passing_touchdowns', 
                'rush_touchdowns', 'attempted_passes'
            ], 
            inplace=True
        )
        cd = cd[[
            'p_id', 'wy', 'game_key', 'abbr',
            'total_yards', 'total_touchdowns', 'interceptions_thrown',
            'quarterback_rating', 'adjusted_yards_per_attempt','points_for',
            'points_against', 'won','pass_attempts_perc',
            'completion_percentage'
        ]]
        return cd
    def getTrain_rb(self, cd: pd.DataFrame):
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
    def getTrain_wr(self, cd: pd.DataFrame):
        """
        Returns train data for WRs.
        @params:
            cd   - Required  : position player data (only pg_train cols and/or only 1 wy data for individuals)
        """
        # cd = cd.loc[cd['volume_percentage']>0.1] # filter inactive players
        return cd
    def saveTrainData(self, wy: str):
        """
        Saves train data to perGameTrain folder.
        @params:
            wy   - Required  : determines if train data is save for singular wy or ALL (empty wy)
        """
        isAll = (len(wy) == 0)
        cd = self.cd.copy() if isAll else self.cd.loc[self.cd['wy']==wy]
        cd.reset_index(drop=True, inplace=True)
        cd: pd.DataFrame = cd[['p_id', 'wy', 'game_key', 'abbr']+self.pg_train_cols[self.position]]
        cd = self.train_funcs[self.position](cd)
        fn_wy = 'all' if isAll else wy.replace(' | ','-')
        if not isAll:
            cd.drop(columns=['wy', 'game_key', 'abbr'], inplace=True)
        cd = cd.round(2)
        self.saveFrame(cd, (self.train_dir + self.position + "_train_" + fn_wy))
        print(self.position + "_train_" + fn_wy + ".csv written to " + self.train_dir)
        return
    def buildTargetData(self, wy: str):
        fn_wy = wy.replace(' | ','-')
        train_fn = self.train_dir + self.position + "_train_" + fn_wy
        target_fn = self.target_dir + self.position + "_target_" + fn_wy
        df = pd.read_csv("%s.csv" % train_fn)
        dp = DataFrameDisplay(df)
        grades = dp.radio_values
        new_df = df[['p_id']]
        new_df['grade'] = grades
        new_df.to_csv("%s.csv" % target_fn, index=False)
        print(target_fn + ' saved.')
        return
    def saveTTData(self, wy: str):
        """
        Writes both train and target data to a given directory.
        @params:
            wy   - Required  : week | year for added data
        """
        self.saveTrainData(wy)
        self.buildTargetData(wy)
        return
    def getHeatMap(self):
        train = pd.concat([pd.read_csv(self.train_dir + fn) for fn in os.listdir(self.train_dir) if 'all' not in fn])
        target = pd.concat([pd.read_csv(self.target_dir + fn) for fn in os.listdir(self.target_dir) if 'all' not in fn])
        data: pd.DataFrame = train.merge(target, on=['p_id'])
        corrmat = data.corr()
        k = 20
        cols = corrmat.nlargest(k, 'grade')['grade'].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=0.75)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()
        return
    def getCorrCols(self, data: pd.DataFrame, k):
        corrmat = data.corr()
        cols = list(corrmat.nlargest(k, 'grade')['grade'].index)
        cols.remove('grade')
        return cols
    def saveModel(self):
        train = pd.concat([pd.read_csv(self.train_dir + fn) for fn in os.listdir(self.train_dir) if 'all' not in fn])
        target = pd.concat([pd.read_csv(self.target_dir + fn) for fn in os.listdir(self.target_dir) if 'all' not in fn])
        data: pd.DataFrame = train.merge(target, on=['p_id'])
        best_cols = self.getCorrCols(data, 20)
        with open(self.models_dir + self.position +  '_best_cols.txt', 'w') as f:
            f.write(','.join(best_cols))
        f.close()
        X = data[best_cols]
        y = data['grade']
        all_models = []
        for _ in range(50):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = LinearRegression(n_jobs=-1)
            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)
            all_models.append((model, acc))
        all_models.sort(key=lambda x: x[1], reverse=True)
        best_model: LinearRegression = all_models[0][0]
        print(f"Best model accuracy: {all_models[0][1]}")
        pickle.dump(best_model, open((self.models_dir + self.position + '_model.sav'), 'wb'))
        return
    def setModel(self):
        self.model = pickle.load(open((self.models_dir + self.position + '_model.sav'), 'rb'))
        return
    def setBestCols(self):
        f = open((self.models_dir + self.position + '_best_cols.txt'), 'r')
        self.best_cols = f.read().split(",")
        return
    def testModel(self, wy):
        """
        Prints top-2 best columns along with their predicted grades.
        @params:
            wy   - Required  : determines week being tested.
        """
        self.setModel()
        self.setBestCols()
        cd = self.cd.loc[self.cd['wy']==wy]
        cd.reset_index(drop=True, inplace=True)
        cd: pd.DataFrame = cd[['p_id', 'wy', 'game_key', 'abbr']+self.pg_train_cols[self.position]]
        cd = self.train_funcs[self.position](cd)
        X = cd[self.best_cols]
        preds = self.model.predict(X)
        cd = cd[['p_id'] + self.best_cols[:2]]
        cd['grade'] = preds
        print(cd)
        return
    def predictGrades(self):
        """
        Loads saved model, predicts all per game grades, and writes to data_dir.
        """
        self.setModel()
        self.setBestCols()
        df = pd.read_csv("%s.csv" % (self.train_dir + self.position + "_train_all"))
        preds = self.model.predict(df[self.best_cols])
        df = df[['p_id', 'wy', 'game_key', 'abbr']]
        df['grade'] = preds
        df = df.round(2)
        df.to_csv("%s.csv" % (self.data_dir + self.position + "_gameGrades"), index=False)
        print(f"{self.position}_gameGrades written to data_dir.")
        return
    def buildMockData(self, wy: str):
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
        self.saveFrame(df, (self._dir + "mockGameData_" + wy.replace(" | ","-")))
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
        self.saveFrame(cd, (self._dir + self.position + "_mockPositionData_" + wy.replace(" | ","-")))
        return
    def getTrainData(self, wy: str):
        """
        Returns train data to perGameTrain.
        @params:
            wy   - Required  : determines if train data is save for singular wy
        """
        cd = self.cd.loc[self.cd['wy']==wy]
        cd.reset_index(drop=True, inplace=True)
        cd: pd.DataFrame = cd[['p_id', 'wy', 'game_key', 'abbr']+self.pg_train_cols[self.position]]
        cd = self.train_funcs[self.position](cd)
        cd = cd.round(2)
        return cd
    def update(self):
        """
        Updates game_grades with for unseen wy in gameData/positionData.
        """
        self.setGameGrades()
        self.setBestCols()
        self.setModel()
        last_wy_gg = self.game_grades['wy'].values[-1]
        last_wy_gd = self.df['wy'].values[-1]
        if last_wy_gg != last_wy_gd:
            df = self.getTrainData(last_wy_gd)
            preds = self.model.predict(df[self.best_cols])
            df = df[['p_id', 'wy', 'game_key', 'abbr']]
            df['grade'] = preds
            df = df.round(2)
            self.game_grades = pd.concat([self.game_grades, df])
            self.saveFrame(self.game_grades, (self.data_dir + self.position + "_gameGrades"))
            print(f"{self.position}_gameGrades updated for {last_wy_gd} and written to data_dir.")
        else:
            print(f"{self.position}_gameGrades already up-to-date.")
        return
    def testGameGrades(self, pid: str):
        """
        Display grades for given player.
        Args:
            pid (str): player id
        """
        self.setGameGrades()
        df = self.game_grades.loc[self.game_grades['p_id']==pid]
        print(df)
        return
    def setGameGrades(self):
        self.game_grades = pd.read_csv("%s.csv" % (self.data_dir + self.position + "_gameGrades"))
        return
    # write dataframe
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
    
# / END Models

#########################

# source = pd.read_csv("%s.csv" % "../source/source")
# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")

# df = pd.read_csv("%s.csv" % "../../../../data/gameData")

# m = PgModels('qb', df, './')

# # m.saveTTData('1 | 2022')

# # m.getHeatMap()

# # m.saveModel()

# # m.testModel('10 | 2022')

# m.saveTrainData('')

# m.predictGrades()

# m.testGameGrades('WatsCh00')

# m.buildMockData('1 | 2023')

# m.update()
