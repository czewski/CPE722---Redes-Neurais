#%%
import matplotlib.pyplot
import pandas
import seaborn
import numpy
import warnings
import random

# %% Leitura de dados ===================================================
missing_values = ["?"]
df = pandas.read_csv('../data/cogumelos.csv', sep=',', na_values = missing_values)#, na_values = missing_values

#plt.style.use('ggplot')
colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
#%%
df.head(-3)


# %% Verificando dados faltantes ===================================================
df.describe()
print(df.isnull().sum())
df.info()


# %% Analise exploratória de dados ===================================================

#ordinal and nominal
cat = df[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']].copy()

#cat.info()

#discrete and continuous LITERALLY NO NUMERIC VALUES 
#num = df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].copy()


# %% Distribuição ===================================================

x = df['y'].value_counts().to_list()
colors = ['#D1345B','#34D1BF']
labels = ["Comestível", "Venenoso"]
matplotlib.pyplot.title('Distribuição de Classes')
matplotlib.pyplot.pie(x, labels=labels, autopct="%1.2f%%", colors=colors[::-1], explode=[0, 0.1])

#dataset desbalanceado 
#aplicar SMOTE

# %% Outliers ===================================================
for atri in num:
    matplotlib.pyplot.figure(figsize=(8,4))
    seaborn.boxplot(data=num,x=num[atri],color=random.choice(colors))

#%%
ax = num[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']].plot(kind='box', title='boxplot', showmeans=True)
matplotlib.pyplot.yscale('log')
matplotlib.pyplot.show()
#bastante outlier em duration e pdays, ver oq fazer
#%% counting outliers 
df[df['duration'] > 2600].count()

#%%

# %% Distribuição de númericos ===================================================

for atri in num: 
    matplotlib.pyplot.rc('lines', mew=0, lw=0)
    matplotlib.pyplot.rc('axes', edgecolor='none', grid=False)
    matplotlib.pyplot.title(f"Distribuição de {atri}", fontdict={'fontsize': 14})
    matplotlib.pyplot.hist(num[atri], color=random.choice(colors), align='mid')
    matplotlib.pyplot.show()


#%% 
print(cat['cap-shape'])

# %% Distribuição de categoricos ===================================================

for atri in cat: 
    matplotlib.pyplot.rc('lines', mew=0, lw=0)
    matplotlib.pyplot.rc('axes', edgecolor='none', grid=False)
    matplotlib.pyplot.title(f"Distribuição de {atri}", fontdict={'fontsize': 14})
    matplotlib.pyplot.bar(cat[atri], color=random.choice(colors))
    matplotlib.pyplot.show()


# %% Heatmap numéricos ===================================================
correlation = pandas.DataFrame(num).corr()
print(correlation)
#%%
# Generate a mask for the upper triangle
mask = numpy.triu(numpy.ones_like(correlation, dtype=bool))

# Set up the matplotlib figure
f, ax = matplotlib.pyplot.subplots(figsize=(8, 7))

# Generate a custom diverging colormap
cmap = seaborn.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
seaborn.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)


#%% test this
corr = numpy.corrcoef(numpy.random.randn(10, 200))
mask = numpy.zeros_like(corr)
mask[numpy.triu_indices_from(mask)] = True
with seaborn.axes_style("white"):
    f, ax = matplotlib.pyplot.subplots(figsize=(7, 5))
    ax = seaborn.heatmap(corr, mask=mask, vmax=.3, square=True)

#%%
cm = seaborn.light_palette('red', as_cmap=True)
matplotlib.pyplot.pcolor(correlation, cmap=cm)
matplotlib.pyplot.yticks(numpy.arange(0.5, len(df.index), 1), df.index)
matplotlib.pyplot.xticks(numpy.arange(0.5, len(df.columns), 1), df.columns)
matplotlib.pyplot.show()


# %%
