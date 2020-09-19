import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

dftrain = pd.read_csv("dataset treino.csv")
dftrain.head()

dfpred = pd.read_csv("dataset pred.csv")
dfpred.head()


dftrain.loc[dftrain['sentimento']=='neutro', 'sentimento'] = 0
dftrain.loc[dftrain['sentimento']=='ofensivo', 'sentimento'] = 1
dftrain.loc[dftrain['sentimento']=='positivo', 'sentimento'] = 2
dftrain.loc[dftrain['sentimento']=='antidemocrático', 'sentimento'] = 3

x_train, x_test, y_train, y_test = train_test_split(dftrain.texto, dftrain.sentimento, test_size=0.25)
clf = Pipeline([('vectorizer', CountVectorizer()), ('nb', MultinomialNB())])
clf.fit(x_train.values, y_train.astype('int'))

print(clf.score(x_test, y_test.astype('int')))

pred = dfpred.texto

clf.predict(pred)

dfpred.sentimento = clf.predict(pred)

dfpred.loc[dfpred['sentimento']==0, 'sentimento'] = 'neutro'
dfpred.loc[dfpred['sentimento']==1, 'sentimento'] = 'ofensivo'
dfpred.loc[dfpred['sentimento']==2, 'sentimento'] = 'positivo'
dfpred.loc[dfpred['sentimento']==3, 'sentimento'] = 'antidemocrático'

dfpred.to_csv('dataset classificado.csv')
