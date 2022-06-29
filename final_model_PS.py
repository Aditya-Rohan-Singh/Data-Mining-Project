##################SL Project:Heart Failure Prediction##################
##################Authors: Pradipkumar Rajasekaran and Sonaxy Mohanty##################

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import itertools
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


##############################################PREPROCESSING#############################################
heart = pd.read_csv("heart.csv")

# converting catogerical values into numerical values 
data = heart.copy()
dummies = pd.get_dummies(data,columns=['Sex', 'ChestPainType','RestingECG','ExerciseAngina', 'ST_Slope'])
merged = pd.concat([data, dummies], axis='columns')# Dropping duplicate columns after concate
mergedDup = merged.loc[:,~merged.columns.duplicated()]
X = mergedDup.drop(columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina', 'ST_Slope'])

# Removing 20% of data for testing
Xg = X.drop(columns = ['HeartDisease'] ).copy()
Yg = X['HeartDisease']
X_train, X_test, Y_train, Y_test = train_test_split(Xg,Yg , test_size = 0.2, shuffle = True, random_state = 5 )

# Splitting the leftover training data into  75% training and 25% validation data
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,Y_train, test_size = 0.25, shuffle = True, random_state = 5)

train_data = X_train.join(Y_train)

##########################################RESULT ANALYSIS##########################################################################
def metric(actual, predicted):
    true_positive = [1 for act,pre in zip(actual,predicted) if act == pre == 1 ]
    true_positive = sum(true_positive)
    true_negative = [1 for act,pre in zip(actual,predicted) if act == pre == 0]
    true_negative = sum(true_negative)
    false_positive = [1 for act,pre in zip(actual,predicted) if (act == 0 and pre == 1)]
    false_positive = sum(false_positive)
    false_negative = [1 for act,pre in zip(actual,predicted) if (act == 1 and pre == 0)]
    false_negative = sum(false_negative)
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
    confusion_matrix = np.array([[true_positive, false_positive] , [false_negative, true_negative]])
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = 2 * (precision * recall)/(precision + recall)
    return confusion_matrix, accuracy, precision, recall, f1 

#########################################ROC AND YOUDEN INDEX########################################
def curve_metrics(actual, predicted):
    tp = np.equal(predicted,1) & np.equal(actual,1)
    tn = np.equal(predicted,0) & np.equal(actual,0)
    fp = np.equal(predicted,1) & np.equal(actual,0)
    fn = np.equal(predicted,0) & np.equal(actual,1)
    
    tpr = tp.sum()/(tp.sum()+fn.sum())
    fpr = fp.sum()/(fp.sum()+tn.sum())
    
    return tpr,fpr

def roc(actual, predicted, threshold = 10000):
    roc = np.array([])
    for t in range(threshold + 1):
        p = np.greater_equal(predicted,t/threshold).astype(int)
        tpr,fpr = curve_metrics(actual,p)
        roc = np.append(roc,[fpr,tpr])
    return roc.reshape(-1,2)

def youden_index(actual, predicted, threshold = 10000):
    yi = np.array([])
    for t in range(threshold + 1):
        p = np.greater_equal(predicted,t/threshold).astype(int)
        tpr,fpr = curve_metrics(actual,p)
        diff = tpr - fpr
        yi = np.append(yi,[diff,(t/threshold)])
    return yi.reshape(-1,2)

    
#######################################K NEAREST NEIGHBORS##############################################################    
def euclidean(l1, l2):
        distance = np.sqrt(np.sum(l1-l2)**2)
        return distance 

def train_NN(X_train,Y_train,x_input,k):
    predicted_labels = []

    for predictors in x_input:
        # store the distances 
        dist_arr = []
        for row in range(len(X_train)):
            distances = euclidean(np.array(X_train[row,:]), predictors)
            dist_arr.append(distances)
        dist_arr = np.array(dist_arr)
        
        # array of k nearest neighbours
        kdistances = np.argsort(dist_arr)[:k]
        labels = Y_train[kdistances]
 
        #plurality 
        nearest_label =  mode(labels).mode[0]
        predicted_labels.append(nearest_label)
        
    return predicted_labels


def knn():
    
    X_train_arr = np.array(X_train)
    Y_train_arr = np.array(Y_train)
    X_valid_arr = np.array(X_valid)
    Y_valid_arr = np.array(Y_valid)
    X_test_arr = np.array(X_test)
    Y_test_arr = np.array(Y_test)
    
    k = np.arange(3,30)
    klist = [] #store all the k values
    f1score = [] #store all the f1scores
    for i in k:  
        klist.append(i)
    # Training knn on train set
        Y_pred_train = train_NN(X_train_arr,Y_train_arr,X_train_arr,i) 

    # using validation set to find best k value
        Y_pred_valid = train_NN(X_train_arr, Y_train_arr, X_valid_arr,i)
    
    # Print training results
        print(f"\nWhen k is {i}:\t")
        confusion_matrix, accuracy, precision, recall, f1 = metric(Y_train_arr,Y_pred_train)
        print("TRAINING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    
    # print validation results
        confusion_matrix, accuracy, precision, recall, f1 = metric(Y_valid_arr,Y_pred_valid)
        f1score.append(f1)
        print("VALIDATION:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")\
             
    bestf1 = max(f1score)
    for i,j in zip(f1score,klist):
        if i == bestf1:
            corr_k = j
    print(f"\nBest k-value:{corr_k}")    # for k = 15, getting highest f1_score         
        
    # Appending traning and validation sets
    X_train_valid = np.concatenate((X_train_arr, X_valid_arr))
    Y_train_valid = np.concatenate((Y_train_arr,Y_valid_arr))
    Y_pred_train_valid = train_NN(X_train_valid, Y_train_valid, X_train_valid,15)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_train_valid,Y_pred_train_valid)
    print("\nMERGED DATA(TRAIN AND VALIDATION AFTER HYPERTUNING):\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
   
    # running knn on test set
    Y_test_pred = train_NN(X_train_valid, Y_train_valid, X_test_arr,15)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test_arr,Y_test_pred)
    print("\nTESTING:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

################################################NAIVE BAYES#################################################################################################################
def naive_bayes():
    mergedX = pd.concat([X_train, X_valid])
    mergedY = pd.concat([Y_train, Y_valid])
    
    predict_train  = predictNB(mergedX)
    
    confusion_matrix, accuracy, precision, recall, f1 = metric(mergedY,predict_train)
    print("\nTRAINING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    predict_test = predictNB(X_test)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test,predict_test)
    print("\nTESTING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

# use train_data for training
means = train_data.groupby(["HeartDisease"]).mean()
var = train_data.groupby(["HeartDisease"]).var()
prior = (train_data.groupby("HeartDisease").count() / len(train_data)).iloc[:,1]
classes = np.unique(train_data["HeartDisease"].tolist())

def gaussian(data, mean, var):
    std = np.sqrt(var)
    # probability density function
    pdf = (np.e ** (-0.5 * ((data - mean)/std) ** 2)) / (std * np.sqrt(2 * np.pi))
    return pdf

def predictNB(X):
    predictions = []
    
    for ins in X.index:
        classProbs = []
        inst = X.loc[ins]

        for cla in classes:
            featureProbs = []
            featureProbs.append(np.log(prior[cla]))
            
            for attribute in X_train.columns:
                data = inst[attribute]
                
                mean = means[attribute].loc[cla]
                variance = var[attribute].loc[cla]
            
                probability = gaussian(data, mean, variance)
                # print("\nPROBABILITY\n",probability)

                if probability != 0:
                    probability = np.log(probability)
                else: 
                    probability = 1/len(train_data)

                featureProbs.append(probability)
            
            totProbability = sum(featureProbs)
            classProbs.append(totProbability)

        maxProb = classProbs.index(max(classProbs))
        prediction = classes[maxProb]
        predictions.append(prediction)

    return predictions

###########################################DECISION TREE########################################################################
#Calculating Gini Impurity of the data
def gini_index(y):

    prob= y.value_counts()/y.shape[0]
    gini = 1-np.sum(prob**2)
    return gini

#Calculating Entropy of the data
def entropy(y):
    a = y.value_counts()/y.shape[0]
    entropy = np.sum(-a*np.log2(a+1e-9))
    return entropy

#Calculating variance to aid the calculation for Information Gain of regression data
def variance(y):
    if len(y) == 1:
        return 0
    else:
        return y.var()

#Calculating Information Gain of a variable givena loss function
#Using entropy getting better accuracy
def info_gain(y, split, func=entropy):
    a = sum(split)
    b = split.shape[0] - a

    if a == 0 or b == 0:
        infogain = 0

    else:
        if y.dtypes != 'O': 
            infogain = variance(y) - (a/(a+b) * variance(y[split])) - (b/(a+b) * variance(y[-split]))
        else:
            infogain = func(y) - (a/(a+b) * func(y[split])) - (b/a+b * func(y[-split]))
    
    return infogain

#Creating all possible combinations of a variable to calculate the information gain
def categories(example):
    example = example.unique()
    example_values = []
    for i in range(0, len(example)+1):
        for value in itertools.combinations(example,i):
            value = list(value)
            example_values.append(value)

    return example_values[1:-1]

#Calculating the maximum Information gain in order to find the best split
def max_infogain_split(x,y,func=entropy):
    split_value = []
    infogain = []

    if x.dtypes != 'O':
        numeric_variable = True
    else:
        numeric_variable = False
    
    if numeric_variable:
        variables = x.sort_values().unique()[1:]
    else:
        variables = categories(x)

    #Calculating InfoGain for all values
    for v in variables:
        split = x < v if numeric_variable else x.isin(v)
        ig = info_gain(y, split, func)
        infogain.append(ig)
        split_value.append(v)
    
    if len(infogain) == 0:
        return(None, None, None, False)
    
    else:
        best_infogain = max(infogain)
        best_infogain_index = infogain.index(best_infogain)
        best_split = split_value[best_infogain_index]
        return(best_infogain, best_split, numeric_variable, True) #do we need so many outputs

#Selecting the best split
def best_split(y, data):
    split = data.drop(y, axis= 1).apply(max_infogain_split, y = data[y])
    if sum(split.loc[3,:]) == 0:
        return(None,None,None,None)
    else:
        split = split.loc[:,split.loc[3,:]]
        split_variable = max(split)
        split_value = split[split_variable][1] 
        split_ig = split[split_variable][0]
        split_numeric = split[split_variable][2]
        return(split_variable, split_value, split_ig, split_numeric)

#Making a split based on the best splitting data
def make_split(attri, val, data, numeric):
  
  if numeric:
    data1 = data[data[attri] < val]
    data2 = data[(data[attri] < val) == False]

  else:
    data1 = data[data[attri].isin(val)]
    data2 = data[(data[attri].isin(val)) == False]
  
  return(data1,data2)

#Predicting the data, mean for regression and mode for classification
def predict(data, is_target):
    if is_target:
        pred  = data.value_counts().idxmax()
    
    else:
        pred = data.mean()
    return pred

#Training a decision tree
def train_decisiontree(data,y, is_target, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):
  
  # Check that max_categories is fulfilled
  if counter==0:
    types = data.dtypes
    check_columns = types[types == "object"].index
    for column in check_columns:
      var_length = len(data[column].value_counts()) 
      if var_length > max_categories:
        raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))

  # Check for depth conditions
  if max_depth == None:
    depth_cond = True

  else:
    if counter < max_depth:
      depth_cond = True

    else:
      depth_cond = False

  # Check for sample conditions
  if min_samples_split == None:
    sample_cond = True

  else:
    if data.shape[0] > min_samples_split:
      sample_cond = True

    else:
      sample_cond = False

  # Check for ig condition
  if depth_cond & sample_cond:
    var,val,ig,var_type = best_split(y, data)

    # If ig condition is fulfilled, make split 
    if ig is not None and ig >= min_information_gain:
      counter += 1
      left,right = make_split(var, val, data,var_type)

      # Creating a sub-tree
      split_type = "<=" if var_type else "in"
      question =   "{} {}  {}".format(var,split_type,val)
      subtree = {question: []}

      # Tree being branched based on conditions satisfiability
      yes_answer = train_decisiontree(left,y, is_target, max_depth,min_samples_split,min_information_gain, counter)
      no_answer = train_decisiontree(right,y, is_target, max_depth,min_samples_split,min_information_gain, counter)

      if yes_answer == no_answer:
        subtree = yes_answer

      else:
        subtree[question].append(yes_answer)
        subtree[question].append(no_answer)

    # If it doesn't match IG condition, make prediction
    else:
      pred = predict(data[y],is_target)
      return pred

   # Drop dataset if doesn't match depth or sample conditions
  else:
    pred = predict(data[y],is_target)
    return pred

  return subtree

#Classifying data
def classify_data(sample, decisiontree):
  question = list(decisiontree.keys())[0] 

  if question.split()[1] == '<=':

    if sample[question.split()[0]] <= float(question.split()[2]):
      answer = decisiontree[question][0]
    else:
      answer = decisiontree[question][1]

  else:

    if sample[question.split()[0]] in (question.split()[2]):
      answer = decisiontree[question][0]
    else:
      answer = decisiontree[question][1]

  # If the answer is not a dictionary
  if not isinstance(answer, dict):
    return answer
  else:
    residual_tree = answer
    return classify_data(sample, answer)

#Predicting using the decision tree
def decisionTreePredictions(dataFrame, decisionTree):
    predictions = dataFrame.apply(classify_data, axis = 1, args = (decisionTree,))
    return predictions


def decision_tree():
    #Data
    # Removing 20% of data for testing
    X_train, X_test = train_test_split(data, test_size = 0.2, shuffle = True, random_state = 5 )
    
    # Slitting the leftover training data into  75% training and 25% validation data
    X_train, X_valid = train_test_split(X_train, test_size = 0.25, shuffle = True, random_state = 5)
    
    #Merging training and validation set
    mergedX = pd.concat([X_train, X_valid])
    
    ########################################
    min_samples_split = 20
    min_information_gain  = 1e-5
    max_depth = 6
    #########################################
    
    #fitting dataset
    decisions = train_decisiontree(X_train,'HeartDisease',True, max_depth,min_samples_split,min_information_gain)
    
    #prediction on validation data
    decisionTreeValidResults = decisionTreePredictions(X_valid, decisions)
    confusion_matrix, accuracy, precision, recall, f1 = metric(X_valid.iloc[:, -1],decisionTreeValidResults)
    print("\nTRAINING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    
    #prediction on merged dataset of validation and training
    decisionTreeMergedResults = decisionTreePredictions(mergedX, decisions)
    confusion_matrix, accuracy, precision, recall, f1 = metric(mergedX.iloc[:, -1],decisionTreeMergedResults)
    print("\nMERGED DATA(TRAIN AND VALIDATION AFTER HYPERTUNING):",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    
    #prediction on test data set
    decisionTreeTestResults = decisionTreePredictions(X_test, decisions)
    confusion_matrix, accuracy, precision, recall, f1 = metric(X_test.iloc[:, -1],decisionTreeTestResults)
    print("\nTESTING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
   
##############################################LOGISTIC REGRESSION########################################################################    
def sigmoidFunc(z):
    activation = 1/(1 + np.exp(-z))
    return activation

def lossFunc(actual, predicted):
    loss = -np.mean(actual * (np.log(predicted)) - (1-actual) * np.log(1-predicted))
    return loss

def gradientDescent(X,y, predicted):
    samples = X.shape[0]
    del_weight = (1/samples) * np.dot(X.T, (predicted -y))
    del_bias = (1/samples) * np.sum((predicted - y))
    return del_weight, del_bias

def train_lr(X,y,iterations,lr):
    m,n  = X.shape
    weight = np.zeros(X.shape[1])
    bias = 0
    X = normalize(X)
    loss = []
    for i in range(iterations):
        predicted = sigmoidFunc(np.dot(X, weight) + bias)
        del_weight, del_bias = gradientDescent(X,y, predicted)
        
        # updating parameters
        weight = weight - lr * del_weight
        bias = bias - lr * del_bias
        
        #calciulating loss
        e = lossFunc(y, sigmoidFunc(np.dot(X, weight) + bias)) 
        loss.append(e)        
    return weight, bias, loss

#normalizing X because we are getting nan values for losses
def normalize(X):
    m,n = X.shape
    for i in range(n):
        X = (X - X.mean(axis = 0))/X.std(axis = 0) #how far is X from mean in standard deviation units - stats
    return X
       
def logistic_regression():
    # lrs = np.arange(0.001,1,step = 0.001) #fixing lr to 0.01
    iterations = 1500
    # for lr in lrs: #first we ran for multiple values and found 0.01 
    
    weight, bias, loss = train_lr(X_train,Y_train,iterations,lr = 0.01)
    X_v = normalize(X_valid)
    pred_train = sigmoidFunc(np.dot(X_v,weight) + bias)
    
    #finding best threshold
    ROC = roc(Y_valid,pred_train)
    plt.scatter(ROC[:,0],ROC[:,1])
    plt.plot([0,1],'--',color='Grey')
    plt.title('ROC Curve for Logistic Regression',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize = 16)
    
    value = youden_index(Y_valid, pred_train)
    YI = value[:,0]
    threshold = value[:,1]
    # best_YI = np.max(YI)
    best_threshold = threshold[np.argmax(YI)]
    print(f"\nBest threshold value:{best_threshold}") #best threshold value = 0.2871
    
    #prediction on validation data
    predicted_train = []
    predicted_train = [1 if i>= 0.2871 else 0 for i in  pred_train]
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_valid,predicted_train)
    print("\nTRAINING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    
    #Merging dataset 
    mergedX = pd.concat([X_train, X_valid])
    mergedY = pd.concat([Y_train, Y_valid])
    
    #prediction on merged data set
    weight, bias, loss = train_lr(mergedX,mergedY,iterations,lr = 0.01)
    X_m = normalize(mergedX)
    pred_merged = sigmoidFunc(np.dot(X_m,weight) + bias)
    predicted_merged = []
    predicted_merged = [1 if i>= 0.2871 else 0 for i in  pred_merged]
    confusion_matrix, accuracy, precision, recall, f1 = metric(mergedY,predicted_merged)
    print("\nMERGED DATA(TRAIN AND VALIDATION AFTER HYPERTUNING):",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    
    #prediction on test set
    weight, bias, loss = train_lr(X_test,Y_test,iterations,lr = 0.01)
    X_t = normalize(X_test)
    pred_test = sigmoidFunc(np.dot(X_t,weight) + bias)
    predicted_test = []
    predicted_test = [1 if i>= 0.2871 else 0 for i in  pred_test]
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test,predicted_test)
    print("\nTESTING:",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")
    
############################################HEURISTICS TEST##########################################################################
            
def heuristicsKNN():
    mergedX = np.concatenate((X_train, X_valid))
    mergedY = np.concatenate((Y_train,Y_valid)) 
    knn = KNeighborsClassifier(n_neighbors = 15 )
    knn.fit(mergedX,mergedY)
    pred_KNN = knn.predict(X_test)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test, pred_KNN)
    print("\nTESTING KNN PRE BUILT:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

def heuristicsNaiveBayes():
     mergedX = pd.concat([X_train, X_valid])
     mergedY = pd.concat([Y_train, Y_valid])
     gnb = GaussianNB()
     gnb.fit(mergedX,mergedY)
     pred_NB = gnb.predict(X_test)
     confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test, pred_NB)
     print("TESTING NAIVE BAYES PRE BUILT:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

def heuristicsDecisionTree():
    mergedX = pd.concat([X_train, X_valid])
    mergedY = pd.concat([Y_train, Y_valid])
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=20, min_impurity_decrease=1e-5)
    dt.fit(mergedX,mergedY)
    pred_DT = dt.predict(X_test)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test, pred_DT)
    print("\nTESTING DECISION TREE PRE BUILT:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

def heuristicsLogisticReg():
    mergedX = pd.concat([X_train, X_valid])
    mergedY = pd.concat([Y_train, Y_valid])
    lr = LogisticRegression(penalty='none',max_iter=1500)
    lr.fit(mergedX,mergedY)
    pred_LR = lr.predict(X_test)
    confusion_matrix, accuracy, precision, recall, f1 = metric(Y_test, pred_LR)
    print("\nTESTING LOGISTIC REGRESSION PRE BUILT:\t",f"\nConfusion Matrix:\t{confusion_matrix}\nAccuracy:\t{accuracy}\nPrecision:\t{precision}\nRecall:\t{recall}\nF1 Score:\t{f1}")

############################################MAIN FUNCTION CALL#################################################################################################    
if __name__ == '__main__':
    knn() 
    heuristicsKNN()
    print("#####################################################################################KNN DONE###############################################################################################\n")
    
    naive_bayes()
    heuristicsNaiveBayes()
    print("#####################################################################################NAIVE BAYES DONE#######################################################################################\n")
    
    decision_tree()
    heuristicsDecisionTree()
    print("#####################################################################################DECISION TREE DONE#####################################################################################\n")
   
    
    logistic_regression()
    heuristicsLogisticReg()
    print("#####################################################################################LOGISTIC REGRESSION DONE###############################################################################\n")
    


#################################################REFERENCES#################################################################################################    

# https://www.askpython.com/python/examples/k-nearest-neighbors-from-scratch
#https://www.geeksforgeeks.org/how-to-convert-categorical-variable-to-numeric-in-pandas/
#https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns
#https://blog.devgenius.io/implementing-na%C3%AFve-bayes-classification-from-scratch-with-python-badd5a9be9c3
#https://medium.com/swlh/decision-tree-implementation-from-scratch-in-python-1cff4c00c71f 
#https://zerowithdot.com/decision-tree/
#https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/
#https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea
##https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2

