import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import random

def box_plot(c_eval, x_tick_labels):

    fig1, ax1 = plt.subplots()   
    ax1.boxplot(c_eval)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_title('C-param')
    ax1.set_ylabel('performance')
    plt.show()

    return



# get data
X = pd.read_csv("./brc_training_data.csv")
y = X["label"]
X = X.drop(columns=["label"])



## In case this does not perform well on your laptop you can also downsample the negative class

"""
choose_random = 50
y_zipped = list(zip(range(0, len(y)), y))
y_positive = [y for y in y_zipped if y[1] == 1]
y_negative = [y for y in y_zipped if y[1] == 0]
y_negative_subsampled = [y for y in y_negative if y[0] in random.sample(range(len(y_negative)), choose_random)]
y_chosen = y_positive + y_negative_subsampled
random.shuffle(y_chosen)
X = np.array(X)[[y[0] for y in y_chosen]]
y = [y[1] for y in y_chosen]

"""

"""
We are using Kfold to get an unbiased estimate of the performance for our chosen hyperparameter setting. 

"""

kf = StratifiedKFold(n_splits=5, random_state = 100, shuffle=True)
hard_margin_eval = []
soft_margin_eval = []


for enum, indeces in enumerate(kf.split(X, y)): 
     train_index, test_index = indeces[0], indeces[1]
     X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
     y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]


     # train SVM
     clf = svm.SVC(C=1e10, class_weight = "balanced", kernel = "rbf")
     clf.fit(X_train, y_train)
     res = clf.predict(X_test)
     hard_margin_eval.append(metrics.accuracy_score(y_test, res))
     print(f" ############ accuracy on split {enum} ############")
     print(f"Accuracy hard margin SVM on split {enum}: {metrics.accuracy_score(y_test, res)}")


     # train soft-margin SVM
     clf = svm.SVC(C=1, class_weight = "balanced", kernel = "rbf")
     clf.fit(X_train, y_train)
     res = clf.predict(X_test)
     soft_margin_eval.append(metrics.accuracy_score(y_test, res))
     print(f"Accuracy soft margin SVM on split {enum}: {metrics.accuracy_score(y_test, res)}")


print("####################################################################")
print(f"Accuracy hard margin SVM on all splits: {np.mean(hard_margin_eval)}")
print(f"Accuracy soft margin SVM on all splits: {np.mean(soft_margin_eval)}")



"""
sklearns GridSearchCV performs internally 5-crossfold validation to asess the quality of a hyperparameter setting. 
Using a nested cross-validation we could also test the quality of this fitting approach against another. 
"""


# grid search
parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 5, 10 , 50]}
svc = svm.SVC(class_weight = "balanced")
clf = GridSearchCV(svc, parameters, cv = 5)
clf.fit(X_train, y_train)


c_split_lists = [clf.cv_results_["mean_test_score"][x:x+4] for x in range(0, len(clf.cv_results_["mean_test_score"]), 4)]
x_tick_labels = ['1', '5', '10' , '50']
box_plot(c_split_lists, x_tick_labels)



print("####################################################################")
# get prediction data
X_predict = pd.read_csv("./brc_predict_candidates.csv")
best_svm_list = []
print(clf.best_estimator_)
for enum, indeces in enumerate(kf.split(X, y)): 
     train_index, test_index = indeces[0], indeces[1]
     X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
     y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]
     best_svm = svm.SVC(C=1, class_weight='balanced', kernel='linear')
     best_svm.fit(X_train, y_train)
     res = best_svm.predict(X_test)
     best_svm_list.append(metrics.accuracy_score(y_test, res))
     print(f"Accuracy best SVM on split {enum}: {metrics.accuracy_score(y_test, res)}")

print("####################################################################")
print(f"Accuracy best SVM on  all splits: {np.mean(best_svm_list)}")
print("####################################################################")


best_svm = svm.SVC(C=1, class_weight='balanced', kernel='linear', probability = True)
best_svm.fit(X, y)
prediction = best_svm.predict(X_predict)

score = best_svm.predict_proba(X_predict)
print(f"Found {int(np.sum(prediction))} possible driver genes")


index_sorted, score_sorted = zip(*sorted(zip(range(0, len(score)), [s[1] for s in score]), key=lambda x: x[1], reverse=True))
print(f"Top 20 indeces: {index_sorted[:20]}")

