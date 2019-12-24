import numpy as np
import math
import random as rand
from collections import Counter
import matplotlib.pyplot as plt
import pickle

TRAIN_STEP = 3
TEST_STEP = 2

def main():
    train = extract_data("20ng-train-stemmed.txt", TRAIN_STEP)
    test = extract_data("20ng-test-stemmed.txt", TEST_STEP)
    """class_list = list(set(test.get('titles')))
    temp_terms = set()
    # Extract the list of term
    for line in train.get('lines'):
        temp_terms.update(line)
    terms = list(temp_terms)
    # Calculate the frequency of term on each document
    train_tf = np.array(list(map(lambda line: np.array(list(map(lambda x: line.count(x) / len(line), terms))), train.get('lines'))))
    num_docs_contain = list(map(lambda x: np.sum(train_tf[:, x]), range(len(terms))))
    count = 0
    total_num_term = len(num_docs_contain)
    for i in range(total_num_term):
        if num_docs_contain[i-count] < 0.02:
            terms.pop(i-count)
            num_docs_contain.pop(i-count)
            train_tf = np.delete(train_tf, i-count, axis=1)
            count += 1
    train_tfidf = calculate_tfidf(terms, train_tf, num_docs_contain, train.get('total'))
    with open('terms.pkl', 'wb') as f:
        pickle.dump(terms, f)
    with open('classList.pkl', 'wb') as fc:
        pickle.dump(class_list, fc)
    np.save('train_tfidf',train_tfidf[0])
    np.save('train_idf',np.array(train_tfidf[1]))
    """
    """
    with open('terms.pkl', 'rb') as fi:
        terms = pickle.load(fi)
    train_tfidf = (np.load('train_tfidf.npy'), np.load('train_idf.npy'))
    euclidean_similar = []
    cosin_similar = []
    for line in test.get('lines'):
        test_tf = np.array(list(map(lambda term: line.count(term), terms)))
        test_tfidf = np.multiply(train_tfidf[1], test_tf)
        euclidean_temp = list(map(lambda x: cal_euclidean_dis(test_tfidf, train_tfidf[0][:,x]),range(len(train_tfidf[0][0]))))
        length_test_tfidf = np.linalg.norm(test_tfidf)
        cosin_temp = list(map(lambda x: cal_cosin_dis(length_test_tfidf,test_tfidf, train_tfidf[0][:,x]), range(len(train_tfidf[0][0]))))
        euclidean_similar.append(euclidean_temp)
        cosin_similar.append(cosin_temp)
    with open('euclid.pkl', 'wb') as f:
        pickle.dump(euclidean_similar, f)
    with open('cosin.pkl', 'wb') as c:
        pickle.dump(cosin_similar, c)
    """
    """
    with open('euclid.pkl', 'rb') as fi:
        euclidean_similar = pickle.load(fi)
    with open('cosin.pkl', 'rb') as f:
        cosin_similar = pickle.load(f)
    with open('classList.pkl', 'rb') as fc:
        class_list = pickle.load(fc)
    random_docs = []
    euclidean_docs = []
    cosin_docs = []
    # With k = 1
    for i in range(test.get('total')):
        r = rand.randrange(train.get('total'))
        random_docs.append((train.get('titles')[r],test.get('titles')[i], train.get('titles')[r] == test.get('titles')[i]))
        r = euclidean_similar[i].index(min(euclidean_similar[i]))
        euclidean_docs.append((train.get('titles')[r],test.get('titles')[i], train.get('titles')[r] == test.get('titles')[i]))
        r = cosin_similar[i].index(max(cosin_similar[i]))
        cosin_docs.append((train.get('titles')[r],test.get('titles')[i], train.get('titles')[r] == test.get('titles')[i]))
    acc = calAcc(class_list, random_docs, euclidean_docs, cosin_docs, test)
    class_acc = acc[0]
    random_acc = acc[1]
    euclidean_acc = acc[2]
    cosin_acc = acc[3]

    print(random_acc)
    print(euclidean_acc)
    print(cosin_acc)

    plotOverall(random_acc, euclidean_acc, cosin_acc)
    plotClassAcc(class_acc, class_list)

    # With others k
    k_random_acc = []
    k_euclidean_acc = []
    k_cosin_acc = []
    k_class_acc = []
    for k in range(5,205,10):
        print("With k = " + str(k) + " the accuracy are below: ")
        random_docs = []
        euclidean_docs = []
        cosin_docs = []
        for i in range(test.get('total')):
            r = [train.get('titles')[rand.randrange(train.get('total'))] for x in range(k)]
            rand_title = Counter(r).most_common(1)[0][0]
            random_docs.append((rand_title, test.get('titles')[i], rand_title == test.get('titles')[i]))

            r = euclidean_similar[i].index(get_knn_value(k, euclidean_similar[i], 'min'))
            euclidean_docs.append((train.get('titles')[r],test.get('titles')[i], train.get('titles')[r] == test.get('titles')[i]))

            r = cosin_similar[i].index(get_knn_value(k, cosin_similar[i], 'max'))
            cosin_docs.append((train.get('titles')[r],test.get('titles')[i], train.get('titles')[r] == test.get('titles')[i]))
        acc = calAcc(class_list, random_docs, euclidean_docs, cosin_docs, test)
        class_acc = acc[0]
        random_acc = acc[1]
        euclidean_acc = acc[2]
        cosin_acc = acc[3]
        k_random_acc.append(random_acc)
        k_euclidean_acc.append(euclidean_acc)
        k_cosin_acc.append(cosin_acc)
        k_class_acc.append(class_acc)
    with open('k_random_acc.pkl','wb') as fr:
        pickle.dump(k_random_acc,fr)
    with open('k_euclidean_acc.pkl','wb') as fe:
        pickle.dump(k_euclidean_acc,fe)
    with open('k_cosin_acc.pkl','wb') as fc:
        pickle.dump(k_cosin_acc,fc)
    with open('k_class_acc.pkl','wb') as fcl:
        pickle.dump(k_class_acc,fcl)
    """
    with open('k_random_acc.pkl','rb') as fr:
        k_random_acc = pickle.load(fr)
    with open('k_euclidean_acc.pkl','rb') as fe:
        k_euclidean_acc = pickle.load(fe)
    with open('k_cosin_acc.pkl','rb') as fc:
        k_cosin_acc = pickle.load(fc)
    with open('k_class_acc.pkl','rb') as fcl:
        k_class_acc = pickle.load(fcl)
    with open('classList.pkl', 'rb') as fca:
        class_list = pickle.load(fca)
    x = range(5,205,10)
    plt.plot(x,k_random_acc,'r',label='Random')
    plt.plot(x,k_euclidean_acc,'b',label='Euclidean Distance')
    plt.plot(x,k_cosin_acc,'g',label='Cosin Similar')
    plt.title('Overall Accuracy')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    colorList = ['#B051CE','#C1FFC1','#794044','#D248A2','#F4E628','#247A9A','#B8677A','#E13509','#2e2323',
                    '#eecfeb','#9161c2','#48ece2','#025f28','#6a9d42','#7b801e','#e17f09','#d8b19b','#2eff47','#00ecff','#aa3aaa','#560cb3']
    y = []
    for i in range(len(class_list)):
        temp = []
        for cl in k_class_acc:
            temp.append(cl['random_acc'][i])
        y.append(temp)
    for i,cl in  enumerate(class_list):
        plt.plot(x,y[i],color=colorList[i],label=cl)
    plt.title('Random Accuracy')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    y = []
    for i in range(len(class_list)):
        temp = []
        for cl in k_class_acc:
            temp.append(cl['euclidean_acc'][i])
        y.append(temp)
    for i,cl in  enumerate(class_list):
        plt.plot(x,y[i],color=colorList[i],label=cl)
    plt.title('Euclidean Accuracy')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    y = []
    for i in range(len(class_list)):
        temp = []
        for cl in k_class_acc:
            temp.append(cl['cosin_acc'][i])
        y.append(temp)
    for i,cl in  enumerate(class_list):
        plt.plot(x,y[i],color=colorList[i],label=cl)
    plt.title('Cosin Accuracy')
    plt.xlabel('K value')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
# Plot overall accuracy
def plotOverall(random_acc, euclidean_acc, cosin_acc):
    plt.bar(range(3), [random_acc,euclidean_acc,cosin_acc],color='#7f6d5f', width=0.25, edgecolor='white')
    plt.xlabel('Similarity Type', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(range(3), ['Random','Euclidean','Cosin Similar'], fontsize=10)
    plt.show()

# Plot class accuracy
def plotClassAcc(class_acc, class_list):
    r1 = range(len(class_list))
    r2 = [x + 0.25 for x in r1]
    r3 = [x + 0.25 for x in r2]
    plt.bar(r1, class_acc['random_acc'], color='#7f6d5f', width=0.25, edgecolor='white', label='Random')
    plt.bar(r2, class_acc['euclidean_acc'], color='#557f2d', width=0.25, edgecolor='white', label='Euclidean Distance')
    plt.bar(r3, class_acc['cosin_acc'], color='#3434eb', width=0.25, edgecolor='white', label='Cosin Similar')
    plt.xlabel('Class', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(r2, class_list, fontsize=5)
    plt.legend()
    plt.show()

def calAcc(class_list, random_docs, euclidean_docs, cosin_docs, test):
    random_acc = 0
    euclidean_acc = 0
    cosin_acc = 0
    class_acc = dict()
    class_acc['random_acc'] = [0]*len(class_list)
    class_acc['euclidean_acc'] = [0]*len(class_list)
    class_acc['cosin_acc'] = [0]*len(class_list)
    for i in range(test.get('total')):
        random_acc += random_docs[i][2]/test.get('total')
        euclidean_acc += euclidean_docs[i][2]/test.get('total')
        cosin_acc += cosin_docs[i][2]/test.get('total')
    for k in range(len(class_list)):
        e = class_list[k]
        total = test.get('titles').count(e)
        for i in range(test.get('total')):
            if e == random_docs[i][1]:
                class_acc['random_acc'][k] += random_docs[i][2]/total
            if e == euclidean_docs[i][1]:
                class_acc['euclidean_acc'][k] += euclidean_docs[i][2]/total
            if e == cosin_docs[i][1]:
                class_acc['cosin_acc'][k] += cosin_docs[i][2]/total
    return (class_acc, random_acc, euclidean_acc, cosin_acc)

def extract_data(filePath, step):
    f = open(filePath, "r")
    lines = f.readlines()
    titles = list(map(lambda x: lines[x].split(',')[0], range(0,len(lines),step)))
    content = list(map(lambda x: lines[x].strip().split(",")[1].split(" "), range(0,len(lines),step)))
    return {'titles': titles, 'lines': content, 'total':len(titles)}

def calculate_tfidf(terms, train_tf, num_docs_contain, num_docs):
    idf = list(map(lambda x: math.log(num_docs/x), num_docs_contain))
    tf_idf = list(map(lambda x: np.multiply(idf[x], train_tf[:,x]), range(len(terms))))
    return (np.array(tf_idf),idf)

def get_knn_value(k, values, type):
    sortValue = []
    if type == 'min':
        sortValue = sorted(values)
    elif type == 'max':
        sortValue = sorted(values,reverse=True)
    top_k = Counter(sortValue[:k])
    return top_k.most_common(1)[0][0]

def cal_euclidean_dis(arr1, arr2):
    return np.linalg.norm(arr1-arr2)

def cal_cosin_dis(len_arr1,arr1, arr2):
    return np.dot(arr1, arr2)/(len_arr1*np.linalg.norm(arr2))

if __name__ == '__main__':
    main()
