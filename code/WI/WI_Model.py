# Importation des bibliothèques
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
import pickle

# Importation Dataset
df = pd.read_csv('code/WI/dataset_WI.csv')

Gouvernorat_de_résidence = pd.get_dummies(df['Gouvernorat_de_résidence'], prefix='Gouvernorat_de_résidence')
Bac_Nature = pd.get_dummies(df['Bac_Nature'], prefix='Bac_Nature')
Master_Moy_M1 = pd.cut(x=df['Master_Moy_S1_S2'], bins=[0, 10, 11, 12, 13, 14, 20])
Master_Moy_M2 = pd.cut(x=df['Master_Moy_S3'], bins=[0, 10, 11, 12, 13, 14, 20])
Master_Moy_M1 = pd.get_dummies(Master_Moy_M1, prefix='Master_Moy_S1_S2')
Master_Moy_M2 = pd.get_dummies(Master_Moy_M2, prefix='Master_Moy_S3')

df = pd.concat([Gouvernorat_de_résidence, Bac_Nature, df['Bac_Année'], df['Bac_Moyenne'],df['Licence_Année'],df['Licence_Moy_Informatique'],df['Licence_Moy_Gestion'],df['Licence_Moy_Mathématiques'],df['Licence_Moy_Langues_et_étiques_de_l\'information'],	
                  df['Licence_Moy_L1'], df['Licence_Moy_L2'],df['Licence_Moy_L3'], Master_Moy_M1, Master_Moy_M2], axis=1)


# --------------------------- M1 ------------------------
# X: 
X = pd.concat([Gouvernorat_de_résidence,Bac_Nature,df['Bac_Année'],df['Bac_Moyenne'],df['Licence_Année'],
               df['Licence_Moy_Informatique'],	df['Licence_Moy_Gestion'],df['Licence_Moy_Mathématiques'],	
               df['Licence_Moy_Langues_et_étiques_de_l\'information'],df['Licence_Moy_L1'],	df['Licence_Moy_L2'],	
               df['Licence_Moy_L3']], axis=1)
# y: from sklearn.ensemble import RandomForestClassifier

y = Master_Moy_M1

# Modélisation des données
knn = KNeighborsClassifier(n_neighbors = 63, weights='uniform')

# train
knn.fit(X, y)
# predict
predictions = knn.predict(X)
# accuracy
print("Accuracy = ",accuracy_score(y,predictions))
pickle.dump(knn, open('code/WI/Model_M1_WI.pkl','wb'))


# --------------------------- M2 ------------------------
# X: 
X = pd.concat([Gouvernorat_de_résidence,Bac_Nature,df['Bac_Année'],df['Bac_Moyenne'],df['Licence_Année'],
               df['Licence_Moy_Informatique'],	df['Licence_Moy_Gestion'],df['Licence_Moy_Mathématiques'],	
               df['Licence_Moy_Langues_et_étiques_de_l\'information'],df['Licence_Moy_L1'],	df['Licence_Moy_L2'],	
               df['Licence_Moy_L3']], axis=1)
# y: 
y = Master_Moy_M2

# Modélisation des données
knn = KNeighborsClassifier(n_neighbors = 100, weights='distance')

# train
knn.fit(X, y)
# predict
predictions = knn.predict(X)
# accuracy
print("Accuracy = ",accuracy_score(y,predictions))
pickle.dump(knn, open('code/WI/Model_M2_WI.pkl','wb'))