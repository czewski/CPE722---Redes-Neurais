#%%
import matplotlib.pyplot
import pandas
import seaborn
import numpy
import warnings
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#from xgboost import XGBRegressor
from imblearn.over_sampling import SMOTE

# %% PRÉ-PROCESSAMENTO DE DADOS ===================================================
missing_values = ["?"]
df = pandas.read_csv('../data/cogumelos.csv', sep=',', na_values = missing_values)#, na_values = missing_values

df.head()



#%% drop result (class)
result = df['y']

df.drop(['y'], inplace=True, axis = 1)

#%% encode 2 valued columns
for col in df.columns:
    if len(df[col].value_counts()) == 2:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

df
#%% one hot encoding
# listing down the features that has categorical data

categorial_features = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
# categorial_features = ['job', 'marital', 'poutcome']
for item in categorial_features:
    # assigning the encoded df into a new dfFrame object
    df1 = pandas.get_dummies(df[item], prefix=item)
    df = df.drop(item, axis=1)
    for categorial_feature in df1.columns:
        #Set the new column in df to have corresponding df values
        df[categorial_feature] = df1[categorial_feature]

df.head()

#%%
df.head()


#%% Correlação, precisa checar ainda se é >95
data_corr = df.astype(float).corr()  # used the astype() or else I get empty results

f, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
matplotlib.pyplot.title('Pearson Correlation of Mushroom Features')
# Draw the heatmap using seaborn
seaborn.heatmap(data_corr,linewidths=0.5,vmax=1.0, square=True, annot=True)

#%% FIRST PCA APLLY
print(df.values)
X = df.values
# calling sklearn PCA 
pca = PCA(n_components=2)
# fit X and apply the reduction to X 
x_2d = pca.fit_transform(X)

# Let's see how it looks like in 2D - could do a 3D plot as well
matplotlib.pyplot.figure(figsize = (7,7))
matplotlib.pyplot.scatter(x_2d[:,0],x_2d[:,1], alpha=0.1)
matplotlib.pyplot.show()


# Set a 3 KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0)

# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(x_2d)

LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'g'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
matplotlib.pyplot.figure(figsize = (7,7))
matplotlib.pyplot.scatter(x_2d[:,0],x_2d[:,1], c= label_color, alpha=0.1)
matplotlib.pyplot.show()


#%% kmeans diretao

kmeans = KMeans(n_clusters=2)
kmeans.fit(df)
clusters = kmeans.predict(df)

cluster_df = pandas.DataFrame()
cluster_df['cluster'] = clusters
cluster_df['class'] = result
#seaborn.catplot(col='cluster', y=None, x='class', data=cluster_df,
#         kind='count', order=['p','e'], palette=(["#7d069b","#069b15"]))

#seaborn.catplot(col='cluster', y='cluster', x='class', data=cluster_df,
#        kind="bar", ci=None, order=['p','e'], palette=(["#7d069b","#069b15"]))

#cluster_df['clusters'] = kmeans.labels_
#seaborn.swarmplot(cluster_df['clusters'],cluster_df['cluster'])
 
#%%

#filter rows of original data
filtered_label2 = clusters == 0
 
filtered_label8 = clusters== 1
 
#Plotting the results
matplotlib.pyplot.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'red')
matplotlib.pyplot.scatter(filtered_label8[:,0] , filtered_label8[:,1] , color = 'black')
matplotlib.pyplot.show()

#%% RESULTS
#This DataFrame will allow us to visualize our results.
result_df = pd.DataFrame()

#The column containing the correct class for each mushroom in the test set, 'test_y'.
result_df['test_y'] = np.array(test_y) #(don't wanna make that mistake again!)

#The predictions made by K-Means on the test set, 'test_X'.
result_df['kmeans_pred'] = kmeans_pred
#The column below will tell us whether each prediction made by our K-Means model was correct.
result_df['kmeans_correct'] = result_df['kmeans_pred'] == result_df['test_y']

#The predictions made by Logistic Regression on the test set, 'test_X'.
result_df['logreg_pred'] = logreg_pred
#The column below will tell us whether each prediction made by our Logistic Regression model was correct.
result_df['logreg_correct'] = result_df['logreg_pred'] == result_df['test_y']

#----------------------------------------------------------------------

# %% encode output pra transformar yes/no pra 1/0
df['y'] = LabelEncoder().fit_transform(df['y'])
df['y']

#%%split data n sei se vou precisar, ainda nao fiz analise
y = df['y']
X = df.values # get all columns except the last column

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
