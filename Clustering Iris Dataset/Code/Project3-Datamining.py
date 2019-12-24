from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tr
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from statistics import mean


def main():
    # Load and split dataset
    iris = datasets.load_iris()
    print(iris.keys())
    kf = KFold(n_splits=4,shuffle=True,random_state=1)
    X = iris['data']
    Y = iris['target']
    
    # Linear Discrimination Analysis
    shrinkaList = [x*0.1 for x in range(10)]
    linScoreList = []
    shrinkaScoreLists = []
    for shrinkaVal in shrinkaList:
        print("The shrinkage value is : " + str(shrinkaVal))
        tempLinScoreList = []
        for train_index, test_index in kf.split(X):
            linScore = linDisAna(X[train_index],Y[train_index],X[test_index],Y[test_index],shrinkaVal)
            tempLinScoreList.append(linScore)
            linScoreList.append(tempLinScoreList)
        meanLinScore = mean(tempLinScoreList)
        shrinkaScoreLists.append(meanLinScore)
        print("The LDA average score is : " + str(meanLinScore))
    plt.plot(shrinkaList, shrinkaScoreLists)
    plt.xlabel('Shrinkage Value')
    plt.ylabel('K-fold validation score')
    plt.title('Linear Discrimination Analysis')
    plt.show()
    # k-nearest neighbors
    knnScoreList = []
    neighborScoreList = []
    neighborList = range(3,22,2)
    for neighborVal in neighborList:
        print("The neighbor value is : " + str(neighborVal))
        tempKnnScoreList = []
        for train_index, test_index in kf.split(X):
            knnScore = knnModel(X[train_index],Y[train_index],X[test_index],Y[test_index],neighborVal)
            tempKnnScoreList.append(knnScore)
            knnScoreList.append(tempKnnScoreList)
        meanKnnScore = mean(tempKnnScoreList)
        neighborScoreList.append(meanKnnScore)
        print("The KNN average score is : " + str(meanKnnScore))
    plt.plot(neighborList, neighborScoreList)
    plt.xticks(neighborList)
    plt.xlabel('Neighbor Value')
    plt.ylabel('K-fold validation score')
    plt.title('k-nearest neighbors')
    plt.show()
    # Decision trees
    giniScoreList = []
    entropyScoreList = []
    for train_index, test_index in kf.split(X):
        decScore = decTree(X[train_index],Y[train_index],X[test_index],Y[test_index])
        giniScoreList.append(decScore[0])
        entropyScoreList.append(decScore[1])
    meanGiniScore = mean(giniScoreList)
    meanEntropyScore = mean(entropyScoreList)
    print("The gini decision tree average score is : " + str(meanGiniScore))
    print("The entropy decision tree average score is : " + str(meanEntropyScore))
    k = range(1,5)
    plt.plot(k, linScoreList[shrinkaScoreLists.index(max(shrinkaScoreLists))],'r',label='Linear Discrimination Analysis')
    plt.plot(k, knnScoreList[neighborScoreList.index(max(neighborScoreList))], 'b',label='K Nearest Neighbor')
    plt.plot(k, entropyScoreList, 'g',label='Entropy Decision Tree')
    plt.xlabel('K value')
    plt.ylabel('K-fold validation score')
    plt.title('Performance of 3 models at choosen parameter')
    plt.legend()
    plt.show()
 
def linDisAna(X_train, Y_train, X_test, Y_test, shrinVal):
    # Linear Discriminant Model
    lda = LinearDiscriminantAnalysis(n_components=2,solver='lsqr',store_covariance=False,tol=0.0001,shrinkage=shrinVal)
    # Linear Discriminant learning
    lda.fit(X_train, Y_train)
    sc = lda.score(X_test, Y_test)
    return(sc)

def decTree(X_train, Y_train, X_test, Y_test):
    # Decision Tree Model
    dtGini = DecisionTreeClassifier()
    dtEntropy = DecisionTreeClassifier(criterion='entropy')
    dtGini.fit(X_train, Y_train)
    dtEntropy.fit(X_train, Y_train)
    #Plot decision tree
    tr.plot_tree(dtGini, filled=True)
    plt.show()
    tr.plot_tree(dtEntropy, filled=True)
    plt.show()
    giniTreeScore = dtGini.score(X_test, Y_test)
    entropyTreeScore = dtEntropy.score(X_test, Y_test)
    return (giniTreeScore,entropyTreeScore)

def knnModel(X_train, Y_train, X_test, Y_test, neighbor):
    # KNN Model
    neigh = KNeighborsClassifier(n_neighbors=neighbor)
    neigh.fit(X_train, Y_train)
    neiValue = neigh.score( X_test, Y_test)
    return neiValue

if __name__ == "__main__":
    main()