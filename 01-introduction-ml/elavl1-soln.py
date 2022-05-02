import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# random forest fitting
rfc = RandomForestClassifier(n_estimators=200, random_state=100)
rfc.fit(X_train, y_train)

# random forest metrics
rfc_predictions = rfc.predict(X_test)
print("Metrics for random forest:")
print(f"ROC: {metrics.roc_auc_score(y_test, rfc_predictions)}")
print(f"Accuracy: {metrics.accuracy_score(y_test, rfc_predictions)}")
print(f"Precision: {metrics.precision_score(y_test, rfc_predictions, pos_label='1')}")
print(f"Recall: {metrics.recall_score(y_test, rfc_predictions, pos_label='1')}")
print(f"F1: {metrics.f1_score(y_test, rfc_predictions, pos_label='1')}")

# logistic regression fitting
lr = LogisticRegression()
lr.fit(X_train, y_train)

# logistic regression metrics
lr_predictions = lr.predict(X_test)
print("Metrics for logistic regression:")
print(f"ROC: {metrics.roc_auc_score(y_test, lr_predictions)}")
print(f"Accuracy: {metrics.accuracy_score(y_test, lr_predictions)}")
print(f"Precision: {metrics.precision_score(y_test, lr_predictions, pos_label='1')}")
print(f"Recall: {metrics.recall_score(y_test, lr_predictions, pos_label='1')}")
print(f"F1: {metrics.f1_score(y_test, lr_predictions, pos_label='1')}")
