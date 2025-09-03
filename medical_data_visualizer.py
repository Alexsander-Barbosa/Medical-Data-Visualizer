import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv') # carrega o arquivo csv

# 2
df['imc'] = df['weight'] / ((df['height'] / 100) ** 2) # cria a coluna imc e já calcula o valor
df['overweight'] = 0 # define que os valores iniciais da coluna serão 0
df.loc[df['imc'] > 25, 'overweight'] = 1 # determina que valores de imc >25 serão classificados como 1 em overweight
df = df.drop('imc', axis=1)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)
df['active'] = (df['active'] == 1).astype(int)  # se era 0 vira 1 e se era 1 vira 0

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    g = sns.catplot(
    data=df_cat,
    kind='bar',
    x='variable',
    y='total',
    hue='value',
    col='cardio'
    )


    # 8
    fig = g.fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr(numeric_only=True)

    # 13
    mask = np.triu(corr)



    # 14
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    sns.heatmap(
    corr,
    mask=mask, # Esta linha esconde o triângulo superior
    annot=True,
    fmt='.1f',
    linewidths=.5,
    cmap='viridis',
    cbar_kws={'shrink': 0.7}
    )


    # 16
    fig.savefig('heatmap.png')
    return fig
