import pandas as pd

# Wczytaj dane
df = pd.read_csv('dataset/participants.tsv', sep='\t')

# Zamiana kodów grupy na pełne nazwy (opcjonalne, ale czytelne)
group_map = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}
df['GroupFull'] = df['Group'].map(group_map)

# Dodajemy etykiety binarne
df['group_AD'] = (df['Group'] == 'A').astype(int)
df['group_FTD'] = (df['Group'] == 'F').astype(int)
df['group_CN'] = (df['Group'] == 'C').astype(int)
df['has_dementia'] = df['Group'].isin(['A', 'F']).astype(int)
df['cognitive_decline'] = (df['MMSE'] < 25).astype(int)
df['severe_decline'] = (df['MMSE'] < 18).astype(int)
df['age_over_65'] = (df['Age'] > 65).astype(int)

# Wybierz potrzebne kolumny
labels = df[[
    'participant_id', 'group_AD', 'group_FTD', 'group_CN',
    'has_dementia', 'cognitive_decline', 'severe_decline', 'age_over_65'
]]

# Zapisz jako CSV
labels.to_csv('multi_labels.csv', index=False)

print("Gotowe! Etykiety zapisane w pliku 'multi_labels.csv'")
