#%%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix, plot_confusion_matrix  
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, auc, roc_curve, plot_roc_curve
from sklearn.metrics import  precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from pre_pro import  X_train, X_test, y_train, y_test#, X_val, y_val #sem val por causa do kfold
#from pre_pro import PCA_X_test, PCA_X_train,

import tensorflow as tf
import os #disable logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.regularizers import l2
from keras.backend import clear_session
from ann_visualizer.visualize import ann_viz;
#%% K FOLD CROSS VALIDATION
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
cvscores = []

for train, test in kfold.split(X_train, y_train):
    # create model
    clear_session()
    ann = tf.keras.models.Sequential(name='MLPFullyConnected')
    ann.add(tf.keras.layers.InputLayer(input_shape=(44,)))
    ann.add(tf.keras.layers.Dense(units= 6, activation='relu'))
    ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid')) 
	
    # Compile model
    optimizer = Adam()

    #optimizer = SGD(learning_rate=0.01, decay=1e-5)# , momentum=.9, 

    ann.compile(optimizer=optimizer,
                loss= 'mean_squared_error', 
                metrics= ['accuracy']) #adam

    early_stop = EarlyStopping(monitor='val_loss', 
                            min_delta=0, 
                            patience=5, 
                            verbose=0, 
                            mode='auto')  #testar patiance

    
    ann_history = ann.fit(X_train.iloc[train],y_train[train], 
                        shuffle=True, 
                        use_multiprocessing=True, #validation_split=0.1,
                        batch_size= 128, 
                        epochs= 200, 
                        validation_data = (X_train.iloc[test],y_train[test]),  
                        callbacks=[early_stop])

    scores = ann.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
#%%
print(cvscores)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



#%% KERAS ANN MLP
#maybe add dropout?
clear_session()

ann = tf.keras.models.Sequential(name='MLPFullyConnected')
ann.add(tf.keras.layers.InputLayer(input_shape=(44,)))
ann.add(tf.keras.layers.Dense(units= 6, activation='sigmoid')) #testar tgh
#ann.add(tf.keras.layers.Dense(units= 10, activation='relu')) #testar tgh
ann.add(tf.keras.layers.Dense(units= 1, activation='sigmoid')) #sigmoid

optimizer = Adam()

#testar sgd ainda. 
#optimizer = SGD(learning_rate=0.01, momentum=.9, decay=1e-5,)# 

ann.compile(optimizer=optimizer,
            loss= 'mean_squared_error', 
            metrics= ['accuracy']) #adam

early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0, 
                           patience=5, 
                           verbose=0, 
                           mode='auto')  #testar patiance

ann_history = ann.fit(X_train,y_train, 
                      shuffle=True, 
                      use_multiprocessing=True, #validation_split=0.1,
                      batch_size= 128, 
                      epochs= 200, 
                      validation_data = (X_val, y_val),  
                      callbacks=[early_stop])

scores = ann.evaluate(X_test, y_test, batch_size=128)
#%%
#Loss
loss_train = ann_history.history['loss']
loss_val = ann_history.history['val_loss']
epochs = range(1,len(loss_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#%%
#Acuracia
print("%s: %.2f%%" % (ann.metrics_names[1], scores[1]*100))

ann.summary()
ann.get_config()
loss_train = ann_history.history['accuracy']
loss_val = ann_history.history['val_accuracy']
epochs = range(1,len(loss_train)+1)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% model visualization
from keras.utils.vis_utils import plot_model
plot_model(ann, to_file='model.png')
#%%confusion matrix
y_pred = ann.predict_classes(X_test)
#print(y_pred)
cma = confusion_matrix(y_test, y_pred) #labels='1,2'
#print(cma)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cma.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cma.flatten()/np.sum(cma)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cma, annot=labels, fmt='', cmap='Blues')

#%%
print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {roc_auc_score(y_test, y_pred)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred))
#%%
precision = precision_score(y_test, y_pred)  #Precision Score
recall = recall_score(y_test, y_pred)  #Recall Score
f1 = f1_score(y_test, y_pred)  #F1 Score

print(precision, recall, f1)
#%%
y_test.shape
#%%roc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
sns.set_theme(style = 'white')
plt.figure(figsize = (8, 8))
plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()

#%%error histogram for train, test, validation

sns.histplot(data=ann_history.history['loss'], kde=True, log_scale=True, element='step', fill=False)

# %%
