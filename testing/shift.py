import pandas as pd

data = {
    'epa_dif': [1.345, 0.574, -0.438, 0.873]
}
index = [('A', 'GNB'), ('A', 'DET'),('B', 'RAV'), ('B', 'PIT')]

df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index, names=['key', 'abbr']))
df = df.reset_index()

new_df = pd.DataFrame(columns=['key', 'home_abbr', 'away_abbr', 'home_epa_dif', 'away_epa_dif'])

for i in range(0, len(df.index), 2):
    away = df.iloc[i]
    home = df.iloc[i+1]
    new_df.loc[len(new_df.index)] = [away['key'], home['abbr'], away['abbr'], home['epa_dif'], away['epa_dif']]
    
print(new_df)

