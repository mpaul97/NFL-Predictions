import sys
sys.path.append("../")

import pandas as pd

from paths import DATA_PATH

df = pd.read_csv("%s.csv" % (DATA_PATH + "gameData"))

abbrs = set(list(df['home_abbr'].values))

new_df = pd.DataFrame()
new_df['abbr'] = list(abbrs)

new_df.sort_values(by=['abbr'], inplace=True)

new_df.to_csv("%s.csv" % "teamNames", index=False)