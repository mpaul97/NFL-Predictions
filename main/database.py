import pandas as pd
import numpy as np
import os
import regex as re
import math
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

class Firebase:
    def __init__(self, _dir):
        self._dir = _dir
        self.game_dir = self._dir + "/gamePredictions/"
        self.fan_dir = self._dir + "/fantasyPredictions/"
        self.names_dir = self._dir + "../playerNames_v2/data/"
        self.keys_dir = self._dir + "../api_keys/"
        self.ndf: pd.DataFrame = pd.read_csv("%s.csv" % (self.names_dir + "playerInfo"))
        self.gp: pd.DataFrame = pd.read_csv("%s.csv" % (self.game_dir + "predictions"))
        self.positions = ['QB', 'RB', 'WR', 'TE']
        self.fp: dict = { position: pd.read_csv("%s.csv" % (self.fan_dir + "predictions_" + position)) for position in self.positions }
        self.stat_cols = {
            'QB': [
                'completed_passes', 'attempted_passes', 'passing_yards',
                'passing_touchdowns', 'interceptions_thrown', 'rush_yards',
                'rush_touchdowns'
            ],
            'RB': [
                'rush_yards', 'rush_touchdowns', 'receptions',
                'receiving_yards'
            ],
            'WR': [
                'receptions', 'receiving_yards', 'receiving_touchdowns'
            ],
            'TE': [
                'receptions', 'receiving_yards', 'receiving_touchdowns'
            ]
        }
        self.collections = ['games', 'players', 'qbs', 'rbs', 'wrs', 'tes']
        return
    def get_clean_gp(self, as_dict: bool = True):
        # columns => home_abbr, away_abbr, home_win_probability, away_win_probability, home_points, away_points
        df = self.gp.copy()
        # total win probabilities
        hw_cols = [col for col in df.columns if 'home_won' in col or 'h_won' in col]
        df['total_home_win_probability'] = df.apply(lambda x: np.mean(x[hw_cols]), axis=1)
        hw_prob_cols = [col for col in df.columns if 'home_win_probability' in col]
        df['total_home_win_probability'] = df.apply(lambda x: round(np.mean(x[hw_prob_cols]), 2), axis=1)
        df['total_away_win_probability'] = df['total_home_win_probability'].apply(lambda x: abs(1 - x))
        # points
        hp_cols = [col for col in df.columns if 'home_points' in col]
        df['total_home_points'] = df.apply(lambda x: int(np.mean(x[hp_cols])), axis=1)
        ap_cols = [col for col in df.columns if 'away_points' in col]
        df['total_away_points'] = df.apply(lambda x: int(np.mean(x[ap_cols])), axis=1)
        df = df[['home_abbr', 'away_abbr']+list(df.columns[-4:])]
        df.columns = ['home_team', 'away_team', 'home_win_probability', 'away_win_probability', 'home_points', 'away_points']
        return df.to_dict(orient='records') if as_dict else df
    def get_highest_over(self, row: pd.Series, over_cols: list[str]):
        _dict = row[over_cols].to_dict()
        try:
            max_over = max([int(k.split("_")[1]) for k in _dict if _dict[k] == 1])
            threshold = "Over " + str(max_over) 
        except ValueError:
            threshold = "Under 5"
        return threshold
    def get_name(self, pid: str):
        try:
            return self.ndf.loc[self.ndf['p_id']==pid, 'name'].values[0]
        except IndexError:
            return pid
    def get_clean_fp(self, as_dict: bool = True):
        # columns => name, abbr, position, points, week_rank, threshold projections, stats
        key_cols = ['name', 'abbr', 'position']
        all_cols = key_cols + ['points', 'week_rank', 'threshold_projections']
        all_players = []
        fp_dicts = []
        for position in self.positions:
            df: pd.DataFrame = self.fp[position]
            point_cols = [col for col in df.columns if 'points' in col]
            df['points'] = df.apply(lambda x: round(np.mean(x[point_cols]), 1), axis=1)
            week_cols = [col for col in df.columns if 'week_rank' in col]
            df['week_rank'] = df.apply(lambda x: int(np.mean(x[week_cols])), axis=1)
            df.sort_values(by=['week_rank', 'points'], ascending=[True, False], inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['week_rank'] = df.index + 1
            over_cols = [col for col in df.columns if 'over' in col]
            df['threshold_projections'] = df.apply(lambda x: self.get_highest_over(x, over_cols), axis=1)
            df.sort_values(by=['threshold_projections'], inplace=True)
            for s_col in self.stat_cols[position]:
                stat_cols = [col for col in df.columns if s_col in col]
                df[s_col] = df.apply(lambda x: int(math.ceil(np.mean(x[stat_cols]))), axis=1)
            df['name'] = df['p_id'].apply(lambda x: self.get_name(x))
            df = df[key_cols+all_cols[len(key_cols):]+self.stat_cols[position]]
            df.columns = all_cols + self.stat_cols[position]
            all_players.append(df[all_cols])
            fp_dicts.append(df.to_dict(orient='records') if as_dict else df)
        return fp_dicts + [pd.concat(all_players).to_dict(orient='records') if as_dict else pd.concat(all_players)]
    def delete_and_write(self, db, collection: str, data: dict):
        collection_ref = db.collection(collection)
        # Delete old data
        docs = collection_ref.stream()
        for doc in docs:
            doc.reference.delete()
            print(f"Document: {doc.id} deleted.")
        print(f"Collection {collection} deleted.")
        # Write new data
        [collection_ref.add(d) for d in data]
        print(f"New data written to collection: {collection}.")
        return
    def upload(self):
        # Initialize Firebase with credentials
        fn = [fn for fn in os.listdir(self.keys_dir) if 'nfl-predictions-3e259-firebase' in fn][0]
        cred = credentials.Certificate(self.keys_dir + fn)
        firebase_admin.initialize_app(cred)
        # Get cleaned dicts
        gp = self.get_clean_gp()
        qb, rb, wr, te, players = self.get_clean_fp()
        data = {
            'games': gp, 'players': players, 'qbs': qb,
            'rbs': rb, 'wrs': wr, 'tes': te
        }
        # Create database collections
        db = firestore.client()
        [self.delete_and_write(db, c, data[c]) for c in self.collections]
        print("All data written to Firebase.")
        return
    
######################
    
# fb = Firebase("./")

# fb.upload()

# qbs, rbs, wrs, tes, players = fb.get_clean_fp(as_dict=False)
# players.sort_values(by=['points'], ascending=False, inplace=True)
# players.to_csv("%s.csv" % "players", index=False)