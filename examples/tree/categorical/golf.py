import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Load data from csv
data = pd.read_csv('data/golf.data')

# Separate inputs and target variable
cols_to_retain = ['outlook', 'temperature', 'humidity', 'windy']
X_feature = data[cols_to_retain]
y_train = data['target']

# Encode categorical features
X_feature.loc[:, 'outlook'], outlook_labels = pd.factorize(X_feature['outlook'])
X_feature.loc[:, 'windy'], windy_labels = pd.factorize(X_feature['windy'])

# Create list of features' cardinalities
cardinalities = [len(outlook_labels), -1, -1, len(windy_labels)]

# Train the decision tree
clf = DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X_feature, y_train, cardinalities=cardinalities)

# Export to Graphviz
export_graphviz(clf, out_file='tree.dot',
                feature_names=cols_to_retain,
                class_names=['Don\'t play', 'Play'],
                rounded=True, proportion=False,
                precision=2, filled=True, categories=[outlook_labels, -1, -1, windy_labels])

# Convert to png
from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();
