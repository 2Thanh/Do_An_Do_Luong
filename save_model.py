from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
import os


with open('data/name.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print(LABELS)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

with open('data/model.pkl','wb') as f:
    pickle.dump(knn,f)