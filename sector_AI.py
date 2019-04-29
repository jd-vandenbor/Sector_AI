import pandas as pd
import numpy as np
from nltk.corpus import stopwords

df = pd.read_excel('database.xlsx', index_col=0)
X = df[['Description', 'Primary Industry Sector']]
df.dropna(subset=['Description'], inplace=True)

#------------------------- Make Corpus ----------------------------


X = df['Description']
descriptions = X.tolist()

def make_corpus(text_list):
    from collections import Counter
    words = []
    for item in text_list:
        words += item.split(" ")

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""
        else: words[i] = words[i].lower()

    filtered_words = [word for word in words if word not in stopwords.words('english')]    
    dictionary = Counter(filtered_words)
    del dictionary[""]
    dictionary = dictionary.most_common(3000)
    return dictionary

dictionary = make_corpus(descriptions)
dictionary

#------------------------- Make Features ----------------------------
def make_feature_vectors(dictionary, text_list):

    #set up variables
    feature_set = []
    labels = []
    c = len(text_list) #simple counter

    #go through all documents
    for text in text_list:
        text = text.lower()
        data = []

        #make the feature vector for the document
        words = text.split(" ")
        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)
        
    return feature_set


features = make_feature_vectors(dictionary, descriptions)

#check to see that the feature set matches the number of labels
print(len(features))
features
#-------------------------------------------------------------------
#---- Make Labels ----
L = df['Tech']
labels = L.tolist()


#----------------------Split Data set-----------------------------
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(features, labels, test_size = 0.2, random_state=10) #split data set. We should look at stratifying this


#----------------------Create Naive Bayes model-------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
clf = MultinomialNB()
clf.fit(x_train, y_train)


#----------------------Evaluate model-----------------------------
from sklearn.model_selection import cross_val_score
preds = clf.predict(x_test)
scores = cross_val_score(clf, x_test, y_test, cv=5)
scores  
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_predict

y_scores = cross_val_predict(clf, x_train, y_train, cv=3, method="decision_function") # get decsion scored for all elements in the training set
y_train_pred_90 = (y_scores > 150000) #create new threshold (over 70,000) 
# To make this decision we looked at teh PR graph and noted that if we wanted about an 85% precision we would need a threshold of 150,000.
print(precision_score(y_train, y_train_pred_90)) # == TP / (TP + FP) 
print(recall_score(y_train, y_train_pred_90)) # == TP / (TP + FN)

#plot preciosion recall curve
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# ROC curve (receiver operating characteristic)
# true positive rate versus false positive rate
# dotted line represents a random classifier. A good algorithm stays as far away from that as possible to the top left corner.
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc_curve(fpr, tpr)
plt.show()

# quantify this curve in one number by calculating the area under the curve. Perfect classifier would be = 1
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train, y_scores)









# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import accuracy_score

# clf = SGDClassifier()
# clf.fit(x_train, y_train)
# preds = clf.predict(x_test)
# print(accuracy_score(y_test, preds))

# from sklearn.model_selection import cross_val_score
# #clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, x_test, y_test, cv=5)
# scores  
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





