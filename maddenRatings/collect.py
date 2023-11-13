import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import regex as re
import urllib
from googlesearch import search
from urllib.parse import urljoin
import matplotlib.pyplot as plt
import time

class Collect:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "ratings/"
        self.pids_dir = self._dir + "pids/"
        self.position_dir = self._dir + "../data/positionData/"
        self.tn_df = pd.read_csv("%s.csv" % (self._dir + "../teamNames/teamNames_line"))
        self.tn_df_short = pd.read_csv("%s.csv" % (self._dir + "../teamNames/teamNames"))
        self.pdf = pd.read_csv("%s.csv" % (self._dir + "../playerNames/finalPlayerInfo"))
        self.asp = pd.read_csv("%s.csv" % "../data/allSeasonPlayers")
        self.asp_cols = [col for col in self.asp.columns if col not in ['year', 'abbr']]
        self.target_cols = {
            2002: ['First Name', 'Last Name', 'Overall'],
            2003: ['Name', 'Overall Rating'],
            2004: ['Name', 'Overall'],
            2005: ['FIRSTNAME', 'LASTNAME', 'OVERALLRATING'],
            2006: ['FIRSTNAME', 'LASTNAME', 'OVERALLRATING'],
            2007: ['PLYR_FIRSTNAME', 'PLYR_LASTNAME', 'PLYR_OVERALLRATING'],
            2008: ['First_Name', 'Last_Name', 'Overall_Rating'],
            2009: ['FIRSTNAME', 'LASTNAME', 'OVERALL'],
            2010: ['First', 'Last', 'OVR'],
            2011: ['FIRST NAME', 'LAST NAME', 'OVERALL RATING'],
            2012: ['Name', 'Overall'],
            2013: ['First Name', 'Last Name', 'Overall'],
            2014: ['First Name', 'Last Name', 'Overall'],
            2015: ['FIRST', 'LAST', 'OVERALL RATING'],
            2016: ['First Name', 'Last Name', 'OVR'],
            2017: ['First Name', 'Last Name', 'Overall'],
            2018: ['First Name', 'Last Name', 'Overall'],
            2019: ['Name', 'Overall'],
            2020: ['Name', 'Overall'],
            2021: ['Full Name', 'Overall Rating'],
            2022: ['FirstName', 'LastName', 'OverallRating'],
            2023: ['Full Name', 'Overall Rating'],
            2024: ['Full Name', 'Overall Rating']
        }
        return
    def most_common(self, lst):
        return max(set(lst), key=lst.count)
    def downloadRatings(self, url: str, year: int):
        response = requests.get(url)
        content = response.content
        soup = BeautifulSoup(content, "html.parser")
        a_tags = soup.find_all("a", href=True)
        path = self.data_dir + str(year) + "/"
        os.makedirs(path, exist_ok=True)
        for tag in a_tags:
            href = tag.get("href")
            if href.endswith(".xlsx") or href.endswith(".xls"):
                absolute_url = urljoin(url, href)
                file_name = os.path.basename(absolute_url)
                file_path = os.path.join(path, file_name)
                response = requests.get(absolute_url)
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"Downloaded: {file_name}")
        return
    def getAbbr(self, fn: str):
        for abbr, name in self.tn_df[['abbr', 'name']].values:
            if name.lower() in fn.replace('_', ' '):
                return abbr
        return None
    def cleanRatings(self, year: int):
        path = self.data_dir + str(year) + "/"
        for fn in os.listdir(path):
            abbr = self.getAbbr(fn)
            if abbr:
                new_fn = abbr + '_' + str(year)
                df = pd.read_excel(path + fn)
                self.saveFrame(df, (path + new_fn))
            os.remove(path + fn)
        return
    def storeAll(self):
        """
        Stores and cleans all team ratings as frames.
        2022 Missing files - all player ratings downloaded
        """
        # years = [i for i in range(2002, 2025)]
        years = [2009]
        for year in years:
            if year != 2014:
                if year > 2005:
                    url = 'https://maddenratings.weebly.com/madden-nfl-' + str(year)[-2:] + '.html'
                else:
                    url = 'https://maddenratings.weebly.com/madden-nfl-' + str(year) + '.html'
            else:
                url = 'https://maddenratings.weebly.com/madden-nfl-25.html'
            c.downloadRatings(url, year)
            c.cleanRatings(year)
        return
    def seperate_2022(self):
        """
        All player ratings dowloaded manually (madden_nfl_22_final_roster.xlsx)
        """
        path = self.data_dir + "2022/"
        df = pd.read_excel(path + 'madden_nfl_22_final_roster.xlsx')
        teams = list(set(df['Team']))
        teams.sort()
        tn = self.tn_df_short
        for team in teams:
            abbr = tn.loc[tn['name'].str.contains(team), 'abbr'].values[0]
            temp_df: pd.DataFrame = df.loc[df['Team']==team]
            self.saveFrame(temp_df, (path + abbr + "_2022"))
        return
    def analyze(self):
        years = [i for i in range(2002, 2025)]
        all_abbrs = list(set(self.tn_df['abbr']))
        for year in years:
            path = self.data_dir + str(year) + "/"
            abbrs = [fn.split("_")[0] for fn in os.listdir(path)]
            print(year, set(all_abbrs).difference(set(abbrs)))
        return
    def saveOverallRatings(self):
        df_list = []
        years = [i for i in range(2002, 2025)]
        for year in years:
            print(year)
            path = self.data_dir + str(year) + "/"
            for fn in os.listdir(path):
                print(fn)
                abbr = fn[:3]
                df = pd.read_csv(path + fn)
                cols = self.target_cols[year]
                df = df[cols]
                if len(cols) > 2: # combine first and last names
                    f_col = [col for col in cols if 'first' in col.lower()][0]
                    l_col = [col for col in cols if 'last' in col.lower()][0]
                    names = []
                    for first_name, last_name in df[[f_col, l_col]].values:
                        if year != 2010:
                            names.append(first_name + " " + last_name)
                        else: # remove 2 trailing spaces
                            names.append(first_name + " " + last_name[:-2])
                    df.insert(0, 'name', names)
                    df.drop(columns=[f_col, l_col], inplace=True)
                if year == 2012: # remove column names and NaN in rows
                    df.dropna(inplace=True)
                    df = df.loc[df['Name']!='Name']
                df.columns = ['name', 'overall_rating']
                df.insert(1, 'year', (year - 1))
                df.insert(2, 'abbr', abbr)
                df_list.append(df)
        new_df = pd.concat(df_list)
        self.saveFrame(new_df, "allOverallRatings_01-23")
        return
    def plot(self, names: list[str]):
        df = pd.read_csv("%s.csv" % "allOverallRatings_01-23")
        for name in names:
            temp_df = df.loc[df['name'].str.contains(name)]
            plt.plot(temp_df['year'], temp_df['overall_rating'])
        plt.show()
        return
    def buildAllPids(self):
        df = pd.read_csv("%s.csv" % "allOverallRatings_01-23")
        names = list(set(df['name']))
        ndf = pd.DataFrame(columns=['name', 'p_id'])
        for index, name in enumerate(names):
            self.printProgressBar(index, len(names), 'AllPids')
            try:
                pid = self.pdf.loc[
                    (self.pdf['name'].str.contains(name))|
                    (self.pdf['aka'].str.contains(name)),
                    'p_id'
                ].values[0]
            except IndexError:
                pid = 'UNK'
            ndf.loc[len(ndf.index)] = [name, pid]
        print(len(ndf.loc[ndf['p_id']=='UNK', 'name'].values))
        self.saveFrame(ndf, (self.pids_dir + "allPids"))
        return
    def cleanPids(self):
        df = pd.read_csv("%s.csv" % (self._dir + "allPids"))
        df = df.loc[~df['name'].str.contains('#')]
        self.saveFrame(df, (self._dir + "allPids"))
        df = df.loc[df['p_id']=='UNK']
        self.saveFrame(df, (self.pids_dir + "unkPids"))
        return
    def getUnkPids(self, name, url):
        pattern = r"/players/[A-Z]/.+.htm"
        try: # pfr search works
            fp = urllib.request.urlopen(url)
            mybytes = fp.read()
            mystr = mybytes.decode("utf8", errors='ignore')
            fp.close()
            start = mystr.index('<h1>Search Results</h1>')
            mystr = mystr[start:]
            end = mystr.index('class="search-pagination"')
            mystr = mystr[:end]      
        except ValueError: # pfr search does not work
            all_urls = []
            for i in search(name + ' pro football reference', num=5, stop=5, pause=1):
                if re.search(r"www\.pro-football-reference\.com/players/[A-Z]/", i):
                    all_urls.append(i)
            mystr = '\n'.join(all_urls)
        links = re.findall(pattern, mystr)
        pids = []
        for link in links:
            link = link.split('/')
            pid = link[-1].replace('.htm', '')
            if pid not in pids:
                pids.append(pid)
        return pids
    def saveMissingPids(self):
        df = pd.read_csv("%s.csv" % (self._dir + "unkPids"))
        new_df = pd.DataFrame(columns=['name', 'pids'])
        for index, name in enumerate(df['name'].values):
            self.printProgressBar(index, len(df.index), 'Unk Pids')
            pfr_name = name.lower().replace(' ', '+')
            url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
            try:
                pids = self.getUnkPids(name, url)
            except Exception as e:
                print(f"Error occured: {e} for name: {name}")
                self.saveFrame(new_df, (self._dir + "unkPids"))
                return
            firstName = name.split(" ")[0]
            lastName = name.split(" ")[-1]
            if len(pids) > 1:
                valid_pids = []
                test_pid = lastName[:4] + firstName[:2]
                for pid in pids:
                    if test_pid in pid:
                        valid_pids.append(pid)
                pids = valid_pids
            new_df.loc[len(new_df.index)] = [name, '|'.join(pids)]
            time.sleep(2)
        self.saveFrame(new_df, (self.pids_dir + "missingPids"))
        return
    def saveMissingPids_2(self):
        df = pd.read_csv("%s.csv" % (self.pids_dir + "missingPids"))
        na_df = df.loc[pd.isna(df['pids'])]
        df.dropna(inplace=True)
        df['length'] = df['pids'].apply(lambda x: len(x.split("|")))
        pipe_df = df.loc[df['length']>1]
        pipe_df.drop(columns=['length'], inplace=True)
        self.saveFrame(pd.concat([na_df, pipe_df]), (self.pids_dir + "missingPids_2"))
        df = df.loc[df['length']==1]
        df.drop(columns=['length'], inplace=True)
        df.columns = ['name', 'p_id']
        # self.saveFrame(df, (self.pids_dir + "foundPids"))
        return
    def saveFinalPids(self):
        df = pd.read_csv("%s.csv" % (self._dir + "allPids"))
        df1 = pd.read_csv("%s.csv" % (self._dir + "foundPids"))
        df = df.loc[df['p_id']!='UNK']
        self.saveFrame(pd.concat([df, df1]), (self.pids_dir + "finalPids"))
        return
    def addPidsToRatings(self):
        df = pd.read_csv("%s.csv" % (self._dir + "allOverallRatings_01-23"))
        pdf = pd.read_csv("%s.csv" % (self.pids_dir + "finalPids"))
        pids = []
        for index, name in enumerate(df['name'].values):
            self.printProgressBar(index, len(df.index), 'Adding Pids')
            try:
                pid = pdf.loc[pdf['name']==name, 'p_id'].values[0]
            except IndexError:
                pid = 'UNK'
            pids.append(pid)
        df.insert(1, 'p_id', pids)
        self.saveFrame(df, (self._dir + "allOverallRatings_01-23"))
        return
    def addPositions(self):
        df = pd.read_csv("%s.csv" % (self._dir + "allOverallRatings_01-23"))
        cd = pd.concat([pd.read_csv(self.position_dir + fn) for fn in os.listdir(self.position_dir) if '.csv' in fn])
        positions = []
        for index, pid in enumerate(df['p_id'].values):
            self.printProgressBar(index, len(df.index), 'Adding Positions')
            poses = list(cd.loc[cd['p_id']==pid, 'position'].values)
            positions.append(self.most_common(poses)) if len(poses) != 0 else positions.append('UNK_POS')
        df.insert(2, 'position', positions)
        self.saveFrame(df, (self._dir + "allOverallRatings_01-23"))
        return
    def updateOverallRatings_positions(self):
        """
        Update UNK_POS for rookies
        """
        df = pd.read_csv("%s.csv" % (self._dir + "allOverallRatings_01-23"))
        for index, row in df.iterrows():
            self.printProgressBar(index, len(df.index), 'Updating overallRatings')
            pid, position, abbr, year = row[['p_id', 'position', 'abbr', 'year']]
            if position == 'UNK_POS':
                pdf_info = self.pdf.loc[self.pdf['p_id']==pid, 'position'].values
                new_pos = 'UNK_POS'
                if len(pdf_info) == 0:
                    asp_info = self.asp.loc[(self.asp['abbr']==abbr)&(self.asp['year']==year)]
                    for col in self.asp_cols:
                        col_vals = asp_info[col].values[0]
                        if not pd.isna(col_vals) and pid in col_vals:
                            new_pos = col[:2].upper()
                            break
                else:
                    new_pos = pdf_info[0]
                df.at[index, 'position'] = new_pos
        self.saveFrame(df, (self._dir + "allOverallRatings_01-23"))
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
    
# END / Collect

###########################

c = Collect("./")

# c.storeAll()

# c.seperate_2022()

# c.analyze()

# c.saveOverallRatings()

# c.plot(['Tom Brady', 'Aaron Rodgers'])

# c.buildAllPids()

# c.cleanPids()

# c.saveMissingPids()

# c.saveMissingPids_2()

# c.saveFinalPids()

# c.addPidsToRatings()

# c.addPositions()

c.updateOverallRatings_positions()
