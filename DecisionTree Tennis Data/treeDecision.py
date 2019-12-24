# import numpy package for arrays and stuff 
import numpy as np  
# import matplotlib.pyplot for plotting our result 
import matplotlib.pyplot as plt   
# import pandas for importing csv files  
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

# Function importing Dataset 
def importdata(): 
	balance_data = pd.read_csv('PlayTennis.csv', sep= ',', header = 'infer') 
	
	# Printing the dataswet shape 
	print ("Dataset Length: ", len(balance_data)) 
	print ("Dataset Shape: ", balance_data.shape) 
	
	# Printing the dataset obseravtions 
	print ("Dataset: ",balance_data.head()) 
	return balance_data

def main():
    data = importdata()
    X = data.values[:,0:5]
    Y = data.values[:,5]
    clf_entropy = DecisionTreeClassifier( 
			criterion = "entropy", random_state = 100, 
			max_depth = 3, min_samples_leaf = 5)
    clf_entropy.fit(X, Y)
    dot_data = StringIO()
    export_graphviz(
            clf_entropy,
            out_file =  dot_data,
            feature_names = list(X.columns),
            class_names = 'PlayTennis',
            filled = True,
            rounded = True,
            special_characters = True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('playtennis.png')
    Image(graph.create_png())
    

if __name__ == "__main__":
    main()