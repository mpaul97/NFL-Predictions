import pandas as pd

class Clean:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.raw_dir = self.data_dir + "raw/"
        self.pn_dir = self._dir + "../playerNames_v2/data/"
        self.tn_dir = self._dir + "../teamNames/"
        self.gd_dir = self._dir + "../data/"
        # frames
        self.pn = pd.read_csv("%s.csv" % (self.pn_dir + "playerInfo"))
        self.tn = pd.read_csv("%s.csv" % (self.tn_dir + "propAbbrs"))
        self.cd = pd.read_csv("%s.csv" % (self.gd_dir + "gameData"))
        # info
        self.col_indices = {'QB': 6, 'RB': 6, 'WR': 5}
        self.positions = ['qb', 'rb', 'wr']
        self.simple_markets = {
            'Completions': 'completed_passes', 'Rushing Yards': 'rush_yards',
            'Passing Attempts': 'attempted_passes', 'Touchdown Passes': 'passing_touchdowns',
            'Rushing Attempts': 'rush_attempts', 'Rush & Rec Yards': 'yards_from_scrimmage',
            'Interceptions': 'interceptions_thrown', 'Passing Yards': 'passing_yards',
            'Receptions': 'receptions', 'Receiving Yards': 'receiving_yards'
        }
        return
    def get_abbr(self, prop_abbr: str):
        return self.tn.loc[(self.tn['prop_abbr']==prop_abbr)|(self.tn['alt_prop_abbr']==prop_abbr), 'abbr'].values[0]
    def get_wy(self, week: str):
        year, week = week.split('NFL')
        year = '20' + year
        week = int(week)
        return str(week) + ' | ' + year
    def frames_to_csvs(self):
        frames = pd.read_excel((self.raw_dir + "props.xlsx"), sheet_name=None)
        for key in frames:
            df: pd.DataFrame = frames[key]
            idx = self.col_indices[key]
            new_cols = df.iloc[idx].values
            df = df.loc[df.index>idx]
            df = df.reset_index(drop=True)
            df.dropna(inplace=True)
            df.columns = new_cols
            df = df[['Player', 'Market', 'Team', 'Opp', 'H/A', 'Line', 'Result', 'Week']]
            df.columns = ['player', 'market', 'team', 'opp', 'isHome', 'line', 'result', 'week']
            print(df.shape)
            df['abbr'] = df['team'].apply(lambda x: self.get_abbr(x))
            df['opp_abbr'] = df['opp'].apply(lambda x: self.get_abbr(x))
            df['isHome'] = df['isHome'].apply(lambda x: 1 if x=='Home' else 0)
            df['wy'] = df['week'].apply(lambda x: self.get_wy(x))
            df['home_abbr'] = df.apply(lambda x: x['abbr'] if x['isHome']==1 else x['opp_abbr'], axis=1)
            df['away_abbr'] = df.apply(lambda x: x['opp_abbr'] if x['isHome']==1 else x['abbr'], axis=1)
            cd = self.cd[['key', 'wy', 'home_abbr', 'away_abbr']]
            for index, row in df.iterrows():
                wy, home_abbr, away_abbr = row[['wy', 'home_abbr', 'away_abbr']]
                temp = cd.loc[(cd['wy']==wy)&(cd['home_abbr']==home_abbr)&(cd['away_abbr']==away_abbr)]
                if temp.empty:
                    matchups = cd.loc[cd['wy']==wy, ['home_abbr', 'away_abbr']].values
                    for m in matchups:
                        if home_abbr in m or away_abbr in m:
                            df.at[index, 'home_abbr'] = m[0]
                            df.at[index, 'away_abbr'] = m[1]
            df = cd.merge(df, on=['wy', 'home_abbr', 'away_abbr'])
            print(df.shape)
            df.drop(columns=['home_abbr', 'away_abbr', 'week', 'team', 'opp', 'isHome'], inplace=True)
            df = df[['key', 'wy', 'player', 'abbr', 'opp_abbr', 'market', 'line', 'result']]
            self.save_frame(df, (self.raw_dir + key.lower() + '_data'))
        return
    def get_pid(self, name: str, position: str):
        return self.pn.loc[
            (self.pn['name']==name)&
            (self.pn['positions'].str.contains(position.upper())), 
            'p_id'
        ].values[0]
    def clean(self):
        for pos in self.positions:
            df = pd.read_csv("%s.csv" % (self.raw_dir + pos + "_data"))
            df['p_id'] = df.apply(lambda x: self.get_pid(x['player'], pos), axis=1)
            df['market'] = df['market'].apply(lambda x: self.simple_markets[x])
            df = df[['key', 'wy', 'p_id', 'abbr', 'opp_abbr', 'market', 'line', 'result']]
            self.save_frame(df, (self.raw_dir + pos + '_data'))
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    
# END / Clean

##################

c = Clean("./")

# c.frames_to_csvs()
c.clean()