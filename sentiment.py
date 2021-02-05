#imports

import numpy as np
import tarfile
import nltk
import re
import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.stats import uniform
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import  VotingClassifier

#Download needed
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# Importing drive method from colab for accessing google drive
from google.colab import drive

#Mounting drive
drive.mount('/content/drive')

#TarFile opening and extracting in google drive
file = tarfile.open("/content/drive/My Drive/Colab Notebooks/rt-polaritydata.tar.gz", "r:gz")
file.extractall();

#File opening
def load_doc(filename):
    file = open(filename, encoding='latin-1')
    doc = file.read()
    file.close()
    return doc

#Positive and negative comments
positive = list(open("rt-polaritydata/rt-polarity.pos", "r", encoding="latin-1").readlines())
positive = [s.strip() for s in positive]
negative = list(open("rt-polaritydata/rt-polarity.neg", "r", encoding="latin-1").readlines())
negative = [s.strip() for s in negative]
Total = positive + negative #Storing in the whole in Total


all_word_list=[]
lemmatize = nltk.WordNetLemmatizer()
Stemmer=nltk.stem.PorterStemmer()
stop_words = set(stopwords.words("english"))

#Pre-processing
#Converting in to lower case, words tokenizing, lemmatizing,stemming
for w in np.array(Total):
    w = w.lower()
    w = re.sub("[^a-zA-Z]", " ", str(w))
    w = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", w)
    w = nltk.word_tokenize(w)
    w = [lemmatize.lemmatize(word) for word in w if not word in stop_words]
    w=  [Stemmer.stem(word) for word in w]
    w = " ".join(w)
    all_word_list.append(w)

#Saving features and labels in x and y
x = [sent for sent in all_word_list]
x = np.array(x)
positive_y = [[0, 1] for _ in positive]
negative_y = [[1, 0] for _ in negative]
y = np.concatenate([positive_y, negative_y], 0)



#Converting the data set in to  traning set ,validation set and testing set
x_train,x_test,y_train,y_test = train_test_split(x,np.argmax(y,1),test_size=0.1, random_state=42)
x_train,x_val,y_train,y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

#For GridSearch and Random Search
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_array = X_train_tfidf.toarray()


#GridSearch_Multinobial
param_grid={'alpha':[1,2,3,4,5,6,7,8,9,10]}
clf= GridSearchCV(MultinomialNB(), param_grid)
clf.fit(x_train, y_train)
print(clf.best_params_)

#Multinobial classifier
bayes_clf = Pipeline([("vect", TfidfVectorizer()),
                      ("clf", MultinomialNB(alpha=2.1))])
bayes_clf.fit(x_train, y_train)
predicted = bayes_clf.predict(x_test)
validated = bayes_clf.predict(x_val)
print("Multinomial Naive Bayes validation Accuracy: {:.4f}".format(np.mean(validated == y_val)))
print("Multinomial Naive Bayes Accuracy: {:.4f}".format(np.mean(predicted == y_test)))

#RandomSearch_Logistic
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
clf = RandomizedSearchCV(LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0), distributions)
clf.fit(X_train_array, y_train)
print(clf.best_params_)

#Logistic Classifier
logistic_clf = Pipeline([("vect", TfidfVectorizer()),
                      ("clf", LogisticRegression(C=2.18,penalty='l2',max_iter=200))])
logistic_clf.fit(x_train, y_train)
predicted = logistic_clf.predict(x_test)
validated = logistic_clf.predict(x_val)
print("Logistic Regression validation Accuracy: {:.4f}".format(np.mean(validated == y_val)))
print("Logistic Regression Accuracy: {:.4f}".format(np.mean(predicted == y_test)))

#RandomSearch_SVM
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
clf = RandomizedSearchCV(LinearSVC(), distributions)
clf.fit(x_train, y_train)
print(clf.best_params_)

#SVM Classifier
svm_clf = Pipeline([("vect", TfidfVectorizer()),
                    ("clf",  LinearSVC(C = 0.09,penalty='l2'))])
svm_clf.fit(x_train, y_train)
predicted = svm_clf.predict(x_test)
validated = svm_clf.predict(x_val)
print(" Linear SVC validation Accuracy: {:.4f}".format(np.mean(validated == y_val)))
print("Linear SVC Accuracy: {:.4f}".format(np.mean(predicted == y_test)))

#Voting Classifier
voting_clf = Pipeline([['clf3', VotingClassifier(estimators=  [('lr', logistic_clf),  ('mnb', bayes_clf)], voting='soft')]])
voting_clf.fit(x_train, y_train)
predicted = voting_clf.predict(x_test)
validated = voting_clf.predict(x_val)
print("Voting Classifier validation Accuracy: {:.4f}".format(np.mean(validated == y_val)))
print("voting classifier Accuracy: {:.4f}".format(np.mean(predicted == y_test)))


#Confusion Matrix plot
y_pred = bayes_clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
matrix = plot_confusion_matrix(bayes_clf, x_test, y_test,
                                 normalize='true')
plt.title('Confusion matrix for NaiveBayes classifier')
plt.show(matrix)
plt.show()

