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
df = df.drop('imc', axis=1) # Apaga a coluna imc depois de usa-la para criar a overweight

# 3
# Aqui normaliza, o que era 1 (bom) vira 0 e o que era 2/3 (ruim) vira 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)
df['active'] = (df['active'] == 1).astype(int)  # se era 0 vira 1 e se era 1 vira 0

# 4
def draw_cat_plot():
    # 5
    # usa o método melt que derrete essas colunas, para que temos apenas o concentrado dos valores de cada uma
    # mantem a coluna cardio como referencia para o gráfico
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    # aqui contamos os valores e depois criamos uma coluna com o total
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    # aqui criamos o gráfico de barra com nossos dados tratados
    g = sns.catplot(
    data=df_cat,
    kind='bar',
    x='variable',
    y='total',
    hue='value',
    col='cardio'
    )


    # 8
    # aqui obtemos a figura do gráfico criado anteriormente
    fig = g.fig


    # 9
    # salvamos o arquivo em png (a imagem que aparece no repositorio quando rodamos o main.py)
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    # filtramos nosso dataframe para o intervalo que queremos
    df_heat = df[
    (df['ap_lo'] <= df['ap_hi']) &
    (df['height'] >= df['height'].quantile(0.025)) &
    (df['height'] <= df['height'].quantile(0.975)) &
    (df['weight'] >= df['weight'].quantile(0.025)) &
    (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    # Calcula a matriz de correlação
    corr = df_heat.corr(numeric_only=True)

    # 13
    # gera a mascara para o triangulo superior da matriz
    mask = np.triu(corr)



    # 14
    # cria a figura e define as medidas dela
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15
    # Aqui criamos o mapa de calor usando a matriz e a mascara
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
    # aqui salvamos o mapa criado no item anterior
    fig.savefig('heatmap.png')
    return fig
