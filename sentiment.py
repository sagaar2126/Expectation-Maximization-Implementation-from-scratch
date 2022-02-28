import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score



#TFIDF
nltk.download('stopwords') #For downloading stopwords function.
df=pd.read_csv("sentiment_analysis.csv" , sep=',') 
df.columns=['liked','txt']   #Renaming Columns as mentioned in video.
print(df)
stopset=set(stopwords.words('english'))
vectorizer=TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii',stop_words=stopset)
y=df.liked
X=vectorizer.fit_transform(df.txt)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=42)
clf=naive_bayes.MultinomialNB()
clf.fit(X_train,Y_train)
print(roc_auc_score(Y_test,clf.predict_log_proba(X_test)[:,1])) 



#Bag Of Words.
# vect=CountVectorizer()
# vect.fit(df.txt)

# bag_of_words=vect.transform(df.txt)

vectorizer=CountVectorizer()
y=df.liked
X=vectorizer.fit_transform(df.txt)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,random_state=42)
clf=naive_bayes.MultinomialNB()
clf.fit(X_train,Y_train)
print(roc_auc_score(Y_test,clf.predict_log_proba(X_test)[:,1]))
