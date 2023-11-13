import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import regex as re
import os
from random import randrange
import time
import multiprocessing
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

pd.options.mode.chained_assignment = None

from paths import DATA_PATH

class Converter:
    
    cd = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))
    ndf = pd.read_csv("%s.csv" % "../playerNames/playerNames")
    tndf = pd.read_csv("%s.csv" % "../teamNames/teamNames")
    
    train = pd.DataFrame()
    info = pd.DataFrame()
    vector = pd.DataFrame()
    targets = pd.DataFrame()
    preds = pd.DataFrame()
    
    test_keys = []
    base_dir = ''
    sub_dir = ''
    _dir = ''
    
    def __init__(self, fn, type):
        self.fn = fn
        self.type = type
        self.setDirs()
    
    # set base_dir - scoringSummaries/playByPlay        
    def setDirs(self):
        if self.type == 's':
            self.base_dir = 'scoringSummaries/'
        else:
            self.base_dir = 'playByPlay/'
        self.sub_dir = 'data/' if self.fn == 'all' else 'train/'
        self._dir = self.base_dir + self.sub_dir
        if self.fn != 'all':
            if self.fn not in os.listdir(self._dir):
                os.mkdir(self._dir + self.fn)
            self._dir += self.fn + '/'
        return
    
    # build train - scoringSummaries
    def buildTrainSS(self, keys):
        # make dir if does not exist
        # if self._dir not in os.listdir(self.base_dir + 'train/'):
        #     os.mkdir(self.base_dir + 'train/' + self._dir)
        
        if len(keys) == 0:
            rand_indexes = [randrange(0, len(self.cd.index)) for _ in range(10)]
            cd = self.cd.loc[self.cd.index.isin(rand_indexes)]
        else:
            cd = self.cd.loc[self.cd['key'].isin(keys)]
        
        raw_train = pd.read_csv("%s.csv" % (self.base_dir + "data/rawTrain"))
        
        df_list = []
        
        for index, row in cd.iterrows():
            key = row['key']
            # url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
            # temp_df = self.getContent(url)
            temp_df = raw_train.loc[raw_train['key']==key]
            temp_df['key'] = key
            temp_df['num'] = [i for i in range(len(temp_df.index))]
            df_list.append(temp_df)
            
        df = pd.concat(df_list)
        
        df.reset_index(drop=True, inplace=True)
        
        df = df[['key', 'num', 'Tm', 'Detail', 'away', 'home']]
        
        new_df = pd.DataFrame(columns=['key', 'num', 'detail', 'info'])
    
        for index, row in df.iterrows():
            name = row['Tm']
            detail = row['Detail']
            try:
                abbr = self.tndf.loc[self.tndf['name'].str.contains(name), 'abbr'].values[0]
            except IndexError:
                print(name, '!! MISSING !!')
                return
            # adding info
            home_points = row['home']
            away_points = row['away']
            num = row['num']
            if num == 0:
                home_dif = home_points
                away_dif = away_points
            else:
                home_dif = home_points - df.iloc[index-1]['home']
                away_dif = away_points - df.iloc[index-1]['away']
            dif = home_dif if home_dif != 0 else away_dif
            info = abbr + '|'
            info += 'td|nex' if dif == 6 else ''
            info += 'td|ex' if dif == 7 else ''
            info += 'td|two' if dif == 8 else ''
            info += 'fg' if dif == 3 else ''
            new_df.loc[len(new_df.index)] = [row['key'], num, detail, info]
        
        return new_df
        
    # set train    
    def setTrain(self):
        if self.type == 's':
            if self.fn == 'all':
                self.train = pd.read_csv("%s.csv" % (self._dir +"train"))
            else:
                if len(self.test_keys) != 0:
                    self.train = self.buildTrainSS(self.test_keys)
                else:
                    self.train = pd.read_csv("%s.csv" % (self._dir + self.fn))
        return
    
    # set info from existing   
    def setInfo(self):
        if ('info_' + self.fn + '.csv') in os.listdir(self._dir):
            self.info = pd.read_csv("%s.csv" % (self._dir + 'info_' + self.fn))
        else:
            print(('info_' + self.fn + '.csv') + 'does not exist.')
        return
        
    # names to pids
    def setInfoOverwrite(self):
        
        # if ('info_' + self.fn + '.csv') in os.listdir(self._dir):
        #     overwrite = input(('info_' + self.fn + '.csv') + ' info already exists. Overwrite? (y/n) ')
        #     if overwrite == 'n':
        #         self.info = pd.read_csv("%s.csv" % (self._dir + 'info_' + self.fn))
        #         return
        
        lines = self.train['detail'].values
        
        for index, line in enumerate(lines):
            if self.fn == 'all':
                print(index, len(lines))
            if 'safety' not in lines[index].lower():
                names = self.getNames(lines[index])
                # get pids
                for name in names:
                    try:
                        info = self.ndf.loc[
                            (self.ndf['name'].str.contains(name))|
                            (self.ndf['aka'].str.contains(name)), 
                            ['p_id', 'abbr', 'position']
                        ].values
                        pid = '|'.join(info[0])
                        if info.shape[0] > 1:
                            if 'info' in self.train.columns:
                                team_abbr = (self.train.iloc[index]['info']).split("|")[0]
                            else:
                                team_abbr = self.train.iloc[index]['abbr']
                            for val in info:
                                if val[1] == team_abbr:
                                    pid = '|'.join(val)
                    except IndexError:
                        print(index, name, '!! UNKNOWN !!')
                        return
                    lines[index] = lines[index].replace(name, ('|' + pid + '|'))
            else:
                lines[index] = 'Safety'
                
        self.train['detail'] = lines
        
        self.info = self.train
        
        self.saveInfo()
        
        return
                  
    # converts lines for given df
    def setInfoParallelHelper(self, df):
        
        lines = df['detail'].values
        
        for index, line in enumerate(lines):
            if 'II' in line:
                lines[index] = line.replace('II', '')
            if 'III' in line:
                lines[index] = line.replace('III', '')
            if self.fn == 'all':
                print(index, len(lines))
            if 'safety' not in lines[index].lower():
                names = self.getNames(lines[index])
                # get pids
                for name in names:
                    try:
                        info = self.ndf.loc[
                            (self.ndf['name'].str.contains(name))|
                            (self.ndf['aka'].str.contains(name)), 
                            ['p_id', 'abbr', 'position']
                        ].values
                        pid = '|'.join(info[0])
                        if info.shape[0] > 1:
                            team_abbr = (df.iloc[index]['info']).split("|")[0]
                            for val in info:
                                if val[1] == team_abbr:
                                    pid = '|'.join(val)
                    except IndexError:
                        print(index, name, '!! UNKNOWN !!')
                        return
                    lines[index] = lines[index].replace(name, ('|' + pid + '|'))
            else:
                lines[index] = 'Safety'
                
        df['detail'] = lines
        
        return df
          
    # set info parallel
    def setInfoParallel(self):
        
        if ('info_' + self.fn + '.csv') in os.listdir(self._dir):
            overwrite = input(('info_' + self.fn + '.csv') + ' info already exists. Overwrite? (y/n) ')
            if overwrite == 'n':
                self.info = pd.read_csv("%s.csv" % (self._dir + 'info_' + self.fn))
                return
        
        df_list = []
        
        num_cores = multiprocessing.cpu_count()-1
        num_partitions = num_cores
        df_split = np.array_split(self.train, num_partitions)
        
        if __name__ == 'Converter':
            pool = multiprocessing.Pool(num_cores)
            df_list.append(pd.concat(pool.map(self.setInfoParallelHelper, df_split)))
            pool.close()
            pool.join()
        
        if __name__ == 'Converter':
            if df_list:
                new_df = pd.concat(df_list)
                self.info = new_df
                self.saveInfo()
        
        return
        
    # save info
    def saveInfo(self):
        self.info.to_csv("%s.csv" % (self._dir + 'info_' + self.fn), index=False)
        return
    
    # vectorize
    def vectorize(self):
        
        df = self.info
        
        lines = []
        
        # convert info| to position
        for index, row in df.iterrows():
            line = row['detail']
            line_arr = line.split(" ")
            for i, word in enumerate(line_arr):
                if '|' in word:
                    word_arr = word.split('|')
                    pos = word_arr[-2]
                    line_arr[i] = pos
                if word.isdigit() or word == 'unknown':
                    line_arr[i] = 'number'
            line_arr = [l for l in line_arr if len(l) != 0]
            new_line = ' '.join(line_arr)
            lines.append(new_line)
        
        if self.fn == 'all':
            # vectorize
            cv = CountVectorizer()
            X = cv.fit_transform(lines)
            pickle.dump(cv, open(self.base_dir + 'models/cv.pickle', 'wb'))
        else:
            cv = pickle.load(open(self.base_dir + 'models/cv.pickle', 'rb'))
            X = cv.transform(lines)
        
        new_df = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
        
        new_df.insert(0, 'key', df['key'].values)
        new_df.insert(1, 'num', df['num'].values)
        
        self.vector = new_df
        
        new_df.to_csv("%s.csv" % ((self._dir + 'vector_' + self.fn)), index=False)
        
        return
    
    # collect train from existing all rawTrain
    def collectExisiting(self):
        
        df = pd.read_csv("%s.csv" % (self.base_dir + "/data/rawTrain"))
        
        df = df.loc[df['key'].isin(self.test_keys)]
        
        df.reset_index(drop=True, inplace=True)
    
        df = df[['key', 'num', 'Tm', 'Detail', 'away', 'home']]
        
        new_df = pd.DataFrame(columns=['key', 'num', 'detail', 'info'])
        
        for index, row in df.iterrows():
            name = row['Tm']
            detail = row['Detail']
            try:
                abbr = self.tndf.loc[self.tndf['name'].str.contains(name), 'abbr'].values[0]
            except IndexError:
                print(name, '!! MISSING !!')
                return
            # adding info
            home_points = row['home']
            away_points = row['away']
            num = row['num']
            if num == 0:
                home_dif = home_points
                away_dif = away_points
            else:
                home_dif = home_points - df.iloc[index-1]['home']
                away_dif = away_points - df.iloc[index-1]['away']
            dif = home_dif if home_dif != 0 else away_dif
            info = abbr + '|'
            info += 'td|nex' if dif == 6 else ''
            info += 'td|ex' if dif == 7 else ''
            info += 'td|two' if dif == 8 else ''
            info += 'fg' if dif == 3 else ''
            info += 'sf' if dif == 2 else ''
            new_df.loc[len(new_df.index)] = [row['key'], num, detail, info]
            
        self.train = new_df
        
        return
    
    # set vector from dir
    def setVector(self):
        self.vector = pd.read_csv("%s.csv" % (self._dir + 'vector_' + self.fn))
        return
    
    # set targets from dir
    def setTargets(self):
        self.targets = pd.read_csv("%s.csv" % (self._dir + 'targets_' + self.fn))
        return
    
    # add target from input
    def addTargets(self, targetNames):
        new_df = pd.DataFrame(columns=['key', 'num', 'detail']+targetNames)
        for index, row in self.info.iterrows():
            key = row['key']
            num = row['num']
            detail = row['detail']
            vals = [0 for _ in range(len(targetNames))]
            # for name in targetNames:
                # print(name)
                # val = input('Enter value: ')
                # vals.append(int(val))
            new_df.loc[len(new_df.index)] = [key, num, detail] + vals
        new_df.to_csv("%s.csv" % (self._dir + "targets_" + self.fn), index=False)
        print('Targets created with all 0s. Edit and delete \'detail\'.')
        return
    
    # set preds for testing
    def setPreds(self):
        self.preds = pd.read_csv("%s.csv" % (self._dir + 'preds_' + self.fn))
        return
    
    # save models for each column in targets
    def buildModels(self):
        
        # clear old models
        [os.remove(self.base_dir + 'models/' + fn) for fn in os.listdir(self.base_dir + 'models/') if 'sav' in fn]
        
        data = self.vector.merge(self.targets, on=['key', 'num'], how='left')
        
        X = data.drop(columns=self.targets.columns)
        
        for col in self.targets.columns:
            if col != 'key' and col != 'num':
                y = data[col]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
                model = LogisticRegression(
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)
                fn = self.base_dir + 'models/' + col + '_' + str(round(acc, 2)) + '.sav'
                print(col + " : " + str(round(acc, 2)))
                pickle.dump(model, open(fn, 'wb'))

        return
    
    # save predictions
    def buildPreds(self):
        new_df = pd.DataFrame()
        new_df['key'] = self.vector['key'].values
        new_df['num'] = self.vector['num'].values
        for modelName in os.listdir(self.base_dir + "models/"):
            if '.sav' in modelName:
                model = pickle.load(open((self.base_dir + "models/" + modelName), 'rb'))
                X = self.vector.drop(columns=['key', 'num'])
                preds = model.predict(X)
                new_df[modelName.split("_")[0]] = preds
        new_df.to_csv("%s.csv" % (self._dir + "preds_" + self.fn), index=False)
        self.preds = new_df
        return
    
    # find names
    def getNames(self, line):
        # case 1
        r0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
        # case 2 (Ja'Marr Chase)
        r1 = re.findall(r"[A-Z][a-z]+,?\s?'(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
        # remove case 2 from case 1
        for n0 in r0:
            for n1 in r1:
                if n0 in n1:
                    try:
                        r0.remove(n0)
                    except ValueError:
                        print(n0 + ' already removed.')
        # short first name (J.K.)
        r2 = re.findall(r"[A-Z]\.[A-Z]\.+,?\s(?:[A-Z]*\.?\s*)?[A-Z][a-z]+", line)
        # double capital first name (AJ Dillion)
        r3 = re.findall(r"[A-Z][A-Z]+,?\s?[A-Z][a-z]+", line)
        # capital lower lower capital lower... (CeeDee, JoJo)
        r4 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+,?\s?[A-Z][a-z]+", line)
        # remove case 5 from case 1
        for n0 in r0:
            for n1 in r4:
                if n0 in n1:
                    try:
                        r0.remove(n0)
                    except ValueError:
                        print(n0 + ' already removed.')
        # hyphen last names (Jeremiah Owusu-Koramoah)
        r5 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+?\-[A-Z][a-z]+", line)
        # remove case 6 from case 1
        for n0 in r0:
            for n1 in r5:
                if n0 in n1:
                    try:
                        r0.remove(n0)
                    except ValueError:
                        print(n0 + ' already removed.')
        # hyphen first names (Ray-Ray McCloud)
        r6 = re.findall(r"[A-Z][a-z]+?\-[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
        # remove case 7 from case 1
        for n0 in r0:
            for n1 in r6:
                if n0 in n1:
                    try:
                        r0.remove(n0)
                    except ValueError:
                        print(n0 + ' already removed.')
        # case 8 (D'Ernest Johnson.)
        # r7 = re.findall(r"[A-Z]'[A-Z][a-z]+,?\s(?:[A-Z][a-z]?\s*)?[A-Z][a-z]+", line)
        r7 = re.findall(r"[A-Z]'[A-Z][a-z]+?\s[A-Z][a-z]+", line)
        # remove case 8 from case 1
        for n0 in r0:
            for n1 in r7:
                if n0 in n1:
                    try:
                        r0.remove(n0)
                    except ValueError:
                        print(n0 + ' already removed.')
        # case 9 (Uwe von Schamann)
        r8 = re.findall(r"[A-Z][a-z]+?\s[a-z]+?\s[A-Z][a-z]+", line)
        # remove case 9 from case 1
        for n0 in r0:
            n0_arr = n0.split(" ")
            for word in n0_arr:
                for n8 in r8:
                    if word in n8:
                        try:
                            r0.remove(n0)
                        except ValueError:
                            print(n0 + ' already removed.')
        # case 10 (Neil O'Donnell)
        r9 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?\'[A-Z][a-z]+", line)
        # remove case 10 from case 1
        for n0 in r0:
            n0_arr = n0.split(" ")
            for word in n0_arr:
                for n9 in r9:
                    if word in n9:
                        try:
                            r0.remove(n0)
                        except ValueError:
                            print(n0 + ' already removed.')
        # case 11 (Ka'imi Fairbairn)
        r10 = re.findall(r"[A-Z][a-z]+?\'[a-z]+?\s[A-Z][a-z]+", line)
        # case 12 (Donte' Stallworth)
        r11 = re.findall(r"[A-Z][a-z]+'\s?[A-Z][a-z]+", line)
        # case 13 (J.T. O'Sullivan)
        r12 = re.findall(r"[A-Z].[A-Z].?\s[A-Z]?\'[A-Z][a-z]+", line)
        # # case 14 (Robert Griffin III)
        # r13 = re.findall(r"[A-Z][a-z]+\s?[A-Z][a-z]+\s?[A-Z]+", line)
        # # remove case 14 from case 1
        # for n0 in r0:
        #     for n1 in r13:
        #         if n0 in n1:
        #             try:
        #                 r0.remove(n0)
        #             except ValueError:
        #                 print(n0 + ' already removed.')
        return r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12
    
#############################
        
# c = Converter('', '')

# print(c.getNames("Robert Griffin III kick. Ronald Jones II passed. Donte' Stallworth kick."))
        