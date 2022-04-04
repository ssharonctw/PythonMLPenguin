import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv') #reading the file in same directory


# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy() #place the data of csv into a dataframe
target = 'species'
encode = ['sex','island']


# one hot encoding = Ordinal feature (a.k.a. non-numbers features) encoding
# transform strings values to 0/1 representation
# after running the below: sex (1 col) -> sex_male sex_female (2 col)
#                          island(1 col) -> island_Biscoe, island_Dream, island_Torgarsan (3 col)
# 5 new columns in total (and removing the two orignal columns (sex, island))
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) #prefix = string to append to dataframe column names
    df = pd.concat([df,dummy], axis=1)#concatenate df & dummy horizontally (column concat)
    del df[col] #delete the original column after having the corresponded encoded columns

#turning species string into 0, 1, 2
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and y
X = df.drop('species', axis=1) #all inputs except the species which is our target
Y = df['species']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y) #building the model using x(input) & y(expected output)

# Saving the model using pickle
#(Serializing a python object structure, a.k.a. converting python into a bytestream to store in file)
#'wb' means write content (clf) to file 'penguin_clf.pkl'
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
