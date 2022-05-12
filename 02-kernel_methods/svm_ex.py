import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt


def bar_plot(c_eval, x_tick_labels):

    fig1, ax1 = plt.subplots()   
    ax1.boxplot(c_eval)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_title('C-param')
    ax1.set_ylabel('performance')
    plt.show()

    return



# get data, one list element per entry1
with open("ELAVL1_PARCLIP") as f:
    data_text = f.read().split("\n>")

# list of all possible three-mers
three_mers = [
    f"{a}{b}{c}"
    for a in ["A", "C", "T", "G"]
    for b in ["A", "C", "T", "G"]
    for c in ["A", "C", "T", "G"]
]

data = []
# parse all data elements to get target variable and counts for k-mers
for datum_text in data_text:
    datum = {"target": datum_text.split("\n")[0].split("|")[-1]}
    seq = datum_text.split("\n")[-1]
    for three_mer in three_mers:
        datum[three_mer] = seq.count(three_mer)
    data.append(datum)

# convert to pandas dataframe
df = pd.DataFrame(data)

# do test/train split
X = df.iloc[:, 1:]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=100
)

# train SVM
clf = svm.SVC(C=1e10)
clf.fit(X_train, y_train)
res = clf.predict(X_test)
print(res)
print(f"Accuracy hard margin SVM: {metrics.accuracy_score(y_test, res)}")


# train soft-margin SVM
clf = svm.SVC(C=0.1)
clf.fit(X_train, y_train)
res = clf.predict(X_test)
print(res)
print(f"Accuracy soft margin SVM: {metrics.accuracy_score(y_test, res)}")


# grid search
parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 5, 10 , 15, 20, 25, 30, 35, 40, 45]}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)


c_split_lists = [clf.cv_results_["mean_test_score"][x:x+4] for x in range(0, len(clf.cv_results_["mean_test_score"]), 4)]
x_tick_labels = ['1', '5', '10' , '15', '20', '25', '30', '35', '40', '45']
bar_plot(c_split_lists, x_tick_labels)









