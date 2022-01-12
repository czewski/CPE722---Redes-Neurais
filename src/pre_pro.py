#%%
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
import numpy as np 
#from aed import df

#from sklearn import * 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE

# %% PRÉ-PROCESSAMENTO DE DADOS ===================================================
df = pd.read_csv('../data/bank-full.csv', sep=';')

df.head()
#df = pd.read_csv('../data/bank-full.csv', sep=';')
# %% encode output pra transformar yes/no pra 1/0
df['y'] = LabelEncoder().fit_transform(df['y'])
df['y']

# %% checking the values in education field troca pra -1/1/2/3
df['education'].value_counts()

education_mapper = {"unknown":-1, "primary":1, "secondary":2, "tertiary":3}
df["education"] = df["education"].replace(education_mapper)

#%% drop duration
# data.drop(['duration', 'contact','month','day'], inplace=True, axis = 1)
df.drop(['duration'], inplace=True, axis = 1)

#%%
#need to remove outliers
#balance
df = df[(df['balance']>-6000) & (df['balance']<50000)]

# duration 

#campaign
df = df[(df['campaign']<40)]

#pdays
df = df[(df['pdays']<575)]

#previous
df = df[(df['previous']<50)]

#%%
#binary features
binary_valued_features = ['default','housing', 'loan']
bin_dict = {'yes':1, 'no':0}

#Replace binary values in data using the provided dictionary
for item in binary_valued_features:
    df.replace({item:bin_dict},inplace=True)

    
#%%
#df[['education', 'default', 'housing', 'loan', 'y']].head(-5)

#%% one hot encoding
# listing down the features that has categorical data

categorial_features = ['job', 'marital', 'contact', 'month', 'poutcome']
# categorial_features = ['job', 'marital', 'poutcome']
for item in categorial_features:
    # assigning the encoded df into a new dfFrame object
    df1 = pd.get_dummies(df[item], prefix=item)
    df = df.drop(item, axis=1)
    for categorial_feature in df1.columns:
        #Set the new column in df to have corresponding df values
        df[categorial_feature] = df1[categorial_feature]

df.head()

# %%rearrange the columns in the dataset to contain the y (target/label) at the end
cols = list(df.columns.values)
cols.pop(cols.index('y')) # pop y out of the list
df = df[cols+['y']] #Create new dataframe with columns in new order

#%%split data
y = df['y']
X = df.values[:, :-1] # get all columns except the last column

#smote 11111111111111111111111111111111111111111111
sm = SMOTE(random_state=2)
x_train_smo1, y_train_smo1 = sm.fit_resample(X, y.ravel())

#%%  checking SMOTE progress
print('no SMOTE')
print(df['y'].value_counts())

value = list(y_train_smo1)
print('SMOTE')
print('0', value.count(0))
print('1', value.count(1))



#%% splitting data

# spliting training and testing data #
X_train, X_test, y_train, y_test = train_test_split(x_train_smo1,y_train_smo1, test_size= 0.1, random_state=50)


#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.1, random_state=50)

#%%
# Feature scaling #normalazing? #with_mean=False, with_std=False
scaler = StandardScaler()  #media zero e desvio padrao um
scaler.fit(X)   
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

#split data
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)


#%%
#print(scaler.var_)
#print(X_train)
# X_train.describe()
# #%%
# X_train.mean(axis=0)
# X_train.std(axis=0)
#%%
#%% correlation matrix
correlation_matrix = pd.DataFrame(X_train).corr()
# fig, ax = plt.subplots(figsize=(10,10))         # Sample figsize in inches
# sns.heatmap(correlation_matrix, ax=ax)
# correlation_matrix


#%% removing highly correlated 

# getting the upper triangle of the correlation matrix
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
#print(upper_tri)

# checking which columns can be dropped
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
#print('\nTo drop')
#print(to_drop)

# removing the selected columns
X_train = X_train.drop(X_train.columns[to_drop], axis=1)
X_test = X_test.drop(X_test.columns[to_drop], axis=1)
X_train.to_numpy()
X_test.to_numpy()

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
#%% splitting validation part
#dont need this part if im using kfold. 
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

print('end')
#%% pca?
#ctrl barra group comment

# from sklearn.decomposition import PCA

# # apply the PCA for feature reduction 
# #DONT THINK I NEED THIS 
# pca = PCA(n_components=0.95)
# pca.fit(X_train)
# PCA_X_train = pca.transform(X_train)
# PCA_X_test = pca.transform(X_test)

# X_train


# %%
