
from random import sample
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import pandas as pd
import pickle
from wordcloud import WordCloud, STOPWORDS
from scipy.stats import mode
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


##Functions
def process_tweet(tweet):
    #Remove old style retweet text "RT"
    new_tweet = re.sub(r'^RT[\s]','', tweet)
    new_tweet=re.sub(r'b','',new_tweet)
    #Remove hyperlinks
    new_tweet = re.sub(r'https?:\/\/.*[\r\n]*','', new_tweet)
    #Remove hastags
    new_tweet = re.sub(r'#','',new_tweet)

    #Remove unicode emoji;s
    new_tweet = re.sub(r'((\\x[a-z0-9]{1,}){1,})','', new_tweet)

    # instantiate tokenizer class
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    
    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(new_tweet)    
        
    #Import the english stop words list from NLTK
    stopwords_english = stopwords.words('english') 
    stopwords1 = set(STOPWORDS)
    stopwords1.update(["br", "href","https","t","co","c","b'RT","b'","'","b"])
    
    #Creating a list of words without stopwords
    clean_tweets = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation and word not in stopwords1 :
            clean_tweets.append(word)
        
    #Instantiate Lemmetizer class
    lemmatizer = WordNetLemmatizer()
    
    #Creating a list of lemmetized of words in tweet
    lem_word = []
    for word in clean_tweets:
        lem_word1 = lemmatizer.lemmatize(word)
        lem_word.append(lem_word1)
        # lem_word = lem_word + " " + word
        
    return lem_word


def frequency_builder(sample_train, label_train):
    label_train_list = np.squeeze(label_train).tolist()
  
    freqs = {}
    for y, tweet in zip(label_train_list, sample_train):
        for word in process_tweet(tweet):
                pair = (word, y)
                freqs[pair] = freqs.get(pair, 0) + 1
    return(freqs)

def sigmoid_func(S1,B):
    return 1/(1+np.exp(-S1@B))

def extract_features(tweet, freqs):
    # process_tweet tokenizes, stems, and removes stopwords
    final_tweet = process_tweet(tweet)
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
        
    # loop through each word in the list of words
    for word in final_tweet:
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word,1),0)
        
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word,0),0)
        
    assert(x.shape == (1, 3))
    return x


def gradient_Descent(x, y, B, alpha, num_iters):
    m = len(x)
    for i in range(0, num_iters):
        s = sigmoid_func(x,B)
        B = B - (alpha/m)*np.dot(x.T, s-y)
    return B

def predict_tweet(tweet, freqs, B):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid_func(x,B)
    return y_pred
#-------------------------------------------------

#Load data from kaggle
encoding_used = "ISO-8859-1"
df = pd.read_csv('tweets_training.csv', encoding = encoding_used, header=None)
df.columns = ['label','id','date','query','user','tweet']
#print(df.head())


tweet_data = df.tweet
tweet_label = df.label



#Building training and testing sets
positive = df[df['label'] == 4]
negative = df[df['label'] == 0]


#----------------------------------------------------
#Positive sets
sample_positive=np.array(positive.tweet)
label_positive=np.array(positive.label)
sample_positive = sample_positive[0:35000]
label_positive= label_positive[0:35000]

n1=len(sample_positive)

n_train_pos = 25000#int(n1*0.8)
n_test_pos = 10000#int(n1*0.2)
print("Positive Sample Size:(Train,test)",n_train_pos,n_test_pos)

sample_train_positive=sample_positive[0:n_train_pos]
sample_test_positive=sample_positive[n1-n_test_pos:]

label_train_positive=label_positive[0:n_train_pos]
label_test_positive=label_positive[n1-n_test_pos:]

label_train_positive=np.array(label_train_positive)
label_train_positive=np.array(label_train_positive)

#--------------------------------------------------------
#Negative sets
sample_negative=np.array(negative.tweet)
label_negative=np.array(negative.label)
sample_negative = sample_negative[0:35000]
label_negative= label_negative[0:35000]

n2=len(sample_negative)
print(n2)
n_train_neg = 25000#int(n2*0.8)
n_test_neg = 10000#int(n2*0.2)
print("Negative Sample Size:(Train,test)",n_train_neg,n_test_neg)

sample_train_negative=sample_negative[0:n_train_neg]
sample_test_negative=sample_negative[n2-n_test_neg:]

label_train_negative=label_negative[0:n_train_neg]
label_test_negative=label_negative[n2-n_test_neg:]

label_train_negative=np.array(label_train_negative)
label_test_negative=np.array(label_test_negative)

#------------------------------------------------------------

#Combine positive and negative data set
f1=[sample_train_positive,sample_train_negative]
train_sample=np.concatenate(f1)

f2=[sample_test_positive,sample_test_negative]
test_sample=np.concatenate(f2)

f3=[label_test_positive,label_test_negative]
label_test=np.concatenate(f3)



# combine positive and negative labels
train_y = np.append(np.ones((len(label_train_positive), 1)), np.zeros((len(label_train_negative), 1)), axis=0)
test_y = np.append(np.ones((len(label_test_positive), 1)), np.zeros((len(label_test_negative), 1)), axis=0)

# freqs=frequency_builder(train_sample,train_y)
# print("Frequency Build done")

# with open('saved_frequency.pkl', 'wb') as f:
#     pickle.dump(freqs, f)
with open('saved_frequency.pkl', 'rb') as f:
    freqs = pickle.load(f)

n,p=np.array(train_y).shape
print("Training Labels:",n,p)

n,p=np.array(test_y).shape
print("Testing Labels",n,p)


#Converting training data into features.
# X = np.zeros((len(train_sample), 3))
# for i in range(len(train_sample)):
#     X[i, :]= extract_features(train_sample[i], freqs)
with open('training_features.pkl', 'rb') as f:
    X = pickle.load(f)
Y = train_y
B=np.zeros((3, 1))
alpha=0.0001


#KNN
X_test = np.zeros((len(test_sample), 3))

#-----------------------------------------------------------------
for i in range(len(test_sample)):
    X_test[i, :]= extract_features(test_sample[i], freqs)
    # print(i)


##Write knn code here
ecount = 0
def euclidean(l1, l2):
        distance = np.sqrt(np.sum(l1-l2)**2)
        global ecount
        ecount += 1
        # print("e", ecount)
        return distance 

def train_NN(X_train,Y_train,x_input,k):
    predicted_labels = []
    print("training")

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


def knn(X_train,Y_train,X_test, Y_test):
    
    X_train_arr = np.array(X_train)
    Y_train_arr = np.array(Y_train)
    X_test_arr = np.array(X_test)
    Y_test_arr = np.array(Y_test)
    
    k = 2
    klist = [] #store all the k values
    Y_pred_valid = train_NN(X_train_arr, Y_train_arr, X_test_arr,i)
    print("done")
    mse_test_N=mean_squared_error(test_y,Y_pred_valid)
    print(mse_test_N)

    
    
################################################NAIVE BAYES#################################################################################################################
# probs = []
def naive_bayes(X_train,Y_train,X_test, Y_test):
    with open('training_features.pkl', 'rb') as f:
        X = pickle.load(f)

    df = pd.DataFrame(X, columns=["Bias", "Positive", "Negative"])
    
    predict_train  = predictNB(X_train)
    predict_test = predictNB(X_test)
    sqError = mean_squared_error(Y_test, predict_test, sample_weight=None, multioutput='uniform_average', squared=True)
    print(sqError)

# with open('training_features.pkl', 'rb') as f:
#         X = pickle.load(f)
df1 = pd.DataFrame(train_y, columns=["label"],dtype=float) 

means = pd.to_numeric(df1['label']).groupby(df1['label']).mean()
var = pd.to_numeric(df1['label']).groupby(df1['label']).var()
temp = pd.to_numeric(df1['label']).groupby(df1['label']).count()
prior = (pd.to_numeric(df1['label']).groupby(df1['label']).count() / len(train_sample))#.iloc[:,1]
prior = np.array(prior)

classes = np.unique(df1["label"].tolist())
classes = classes.astype(int)

# model_info ={}
# model_info['means'] = means
# model_info['var'] = var
# model_info['prior'] = prior
# model_info['classes'] = classes

# with open('pickle.pkl','w') as f:
#     pickle.dump(model_info, f)


def gaussian(data, mean, var):
    std = np.sqrt(var)
    # probability density function
    pdf = (np.e ** (-0.5 * ((data - mean)/std) ** 2)) / (std * np.sqrt(2 * np.pi))
    return pdf

def predictNB(X):
    predictions = []
    X = pd.DataFrame(X,columns=["Bias", "Positive", "Negative"])
    for ins in X.index:
        classProbs = []
        inst = X.loc[ins]

        for cla in classes:
            featureProbs = []
            featureProbs.append(np.log(prior[cla]))
            
            for attribute in X.columns:
                print("attribute\n",attribute)
                # This has the column name
                data = inst[attribute]
                print("Attribute\n\n\n\n",attribute,cla)
                mean = means[attribute].loc[cla]
                print(mean,"This place slkfjslfasklgjkgsdh")
                variance = var[attribute].loc[cla]
            
                probability = gaussian(data, mean, variance)
                global probs 
                probs.append(probability)
                # print("\nPROBABILITY\n",probability)

                if probability != 0:
                    probability = np.log(probability)
                else: 
                    probability = 1/len(X)

                featureProbs.append(probability)
            
            totProbability = sum(featureProbs)
            classProbs.append(totProbability)

        maxProb = classProbs.index(max(classProbs))
        prediction = classes[maxProb]
        predictions.append(prediction)

    return predictions


#---------------------------------------------------

# Label_pred = np.array(Label_pred)
test_y = test_y.reshape(-1)
naive_bayes(X,Y,X_test,test_y)

