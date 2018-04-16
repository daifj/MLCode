'''决策树'''
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# 在csv文件中阅读，并将功能添加到命令列表和类标签列表中
allElectronicsData = open('AllElectronics.csv', 'rt')
reader = csv.reader(allElectronicsData)
headers = next(reader)

print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# Vectorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print('dummyX: %s' % dummyX)
print(vec.get_feature_names())

print('labelList: %s' % labelList)

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print('dummyY: %s' % dummyY)

# 使用决策树进行分类。
# Using decision tree for classification
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print('clf: %s' % clf)

# Visualize model 可视化模型
with open('allElectronicInformationGainOri.dot', 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

oneRowX = dummyX[0, :]
print('oneRowX: %s' % oneRowX)

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print('newRowX: %s' % newRowX)

newRowX = newRowX.reshape(1, -1)
predictedY = clf.predict(newRowX)
print('predictedY: %s' % predictedY)
