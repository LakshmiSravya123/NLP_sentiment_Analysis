# NLP_sentiment_Analysis
The main aim of this project is to classify the given mixed review movie corpus in to positive and negative
reviews. In order to achieve this experiments are done on the given data set following difference phases
which includes pre-processing, classification (training and testing). The best machine learning model is
chosen which gives the better performance

# Data sets
rt-polaritydata.tar.gz

# Preprocessing and Feature Extraction
The data set is given in the form of tar-file which includes rt-polarity. neg, rt-polarity.pos text files.
These files are read line by line with the help of sentence tokenization and appended to a list which is
X. The labels Y are stored in the form of one hot vector. Each sentence in the given list X is broken
down into words, lemmatized (eg: Happiness to Happy), stemmed (eg: pulverized to Pulver), converted
in to lower case (eg: HAPPY to happy), removed the special characters and numbering, and removed
stop-words(eg: The, is, that). These words of each sentence are converted to vector format (tfidf) which
gives the weight-age of each word in the document. Top highest frequency words are taken under the
consideration using the max_features parameter.

# Classification
The preprocessed vector is divided into training, test, and validation sets and data is sent to the classifiers
including Naive Bayes, LinearSVC, logistic regression, and voting. During the process of classification,
regularization has been done by tuning the hyperparameters using grid-search and random-search. Fine
tuning has been done and the best classifier is chosen

# Analysis 
According to the given analysis table, Naive Bayes has been tested with the parameter Alpha (1,2,3,....10)
with the help of grid-search giving the optimal value of 2.1.
Logistic regression classifier has been tested with the parameters on random-search C(1,2,...10),penalty(’l2’,
’l1’) Max_iter(100,200,300,....1000) giving the optimal value C=3.1 and Max_iter=500 and penalty(’l2’)
LinearSVC Classifier has been tested with the parameters on random-search C(0, 1,2,...10) and
penalty(’l2’, ’l1’) giving the optimal value C=0.09.
Voting Classifier which includes the previous classifiers (SVM, naive Bayes, and logistic) has been tested
with the parameters on random-search voting(hard, soft) giving the optimal value voting=soft
The best classifier among this is LinearSVC without Stop words. 

It has been observed that stop words have major impact on this data set as stemming and lemmatization
didn’t show much of a difference. Though naivebayes gives the best results in sentimental analysis, because
of the removal of stopwords the there seem to be a difference of of 1 %. Even the range of the hyper
parameters and ngram plays a vital role in the test accuracy

# Results
Classifiers|Parameters| Validation Accuracy (%)| Test Accuracy (%)| Without stop words Test Accuracy(%) | Without Stemming Test Accuracy (%)| Without lemmatization Test Accuracy(%) | Without Stopwords, Stemming , Lemmatization Testing Accuracy (%)|
|------|------|------|-------|-------|-------|-------|---------|
Naive Bayes| Alpha=2.1| 77.28| 76.38|76.4| 75.5|75.9|76.76|
Logistic Regression| C=3.1,Max_iter=500,penalty='12'| 76.1| 75.26|77.6| 74.4|75.54|75.91|
LinearSVC| C=0.09,penalty='12'| 75.9| 75.73|77.32| 74.1|75.73|76.19|
Voting Classifier | voting=soft| 76.88| 75.63|77.04| 75.53|75.91|77.04|

# Languages 
Python (ML classifiers )

