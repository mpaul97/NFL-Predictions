import pandas as pd

data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}
df = pd.DataFrame(data)

cols = ['C', 'D']

def fn(row: pd.Series):
    return { 'C': row['A']+1, 'D': row['B']+1 }

df[cols] = df.apply(lambda row: fn(row), axis='columns', result_type='expand')

print(df)