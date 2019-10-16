import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_graphviz

data = pd.read_csv('data/golf.data')

cols_to_retain = ['outlook', 'temperature', 'humidity', 'windy']

X_feature = data[cols_to_retain]

X_feature['outlook'], outlook_labels = pd.factorize(X_feature['outlook'])
X_feature['windy'], windy_labels = pd.factorize(X_feature['windy'])

feature_mask = [True, False, False, True]

y_train = data['target']
# X_encoded = le.fit_transform(X_feature)

from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=2)
clf = clf.fit(X_feature,y_train, feature_mask=feature_mask)
export_graphviz(clf, out_file='tree.dot',
                feature_names = cols_to_retain,
                class_names = ['Don\'t play', 'Play'],
                rounded = True, proportion = False,
                precision = 2, filled = True, categories=clf.oe.categories_)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in python
import matplotlib.pyplot as plt
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();

