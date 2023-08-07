import pandas as pd

df = pd.read_csv('data/faceDB.csv')

if df.loc[df['name'] == 'kairi', 'desc'].values[0] is None:
    print("None")
print(type(df.loc[df['name'] == 'kairi', 'desc'].values[0]))
print(df.loc[df['name'] == 'kairi', 'desc'].values[0])
