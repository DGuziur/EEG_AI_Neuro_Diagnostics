import pandas as pd

df = pd.read_csv('dataset/participants.tsv', sep='\t')

group_map = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}
df['GroupFull'] = df['Group'].map(group_map)

df['group_AD'] = (df['Group'] == 'A').astype(int)
df['group_FTD'] = (df['Group'] == 'F').astype(int)
df['group_CN'] = (df['Group'] == 'C').astype(int)
df['has_dementia'] = df['Group'].isin(['A', 'F']).astype(int)
df['cognitive_decline'] = (df['MMSE'] < 25).astype(int)
df['severe_decline'] = (df['MMSE'] < 18).astype(int)
df['age_over_65'] = (df['Age'] > 65).astype(int)

labels = df[[
    'participant_id', 'group_AD', 'group_FTD', 'group_CN',
    'has_dementia', 'cognitive_decline', 'severe_decline', 'age_over_65'
]]

labels.to_csv('multi_labels.csv', index=False)

print("Gotowe! Etykiety zapisane w pliku 'multi_labels.csv'")
