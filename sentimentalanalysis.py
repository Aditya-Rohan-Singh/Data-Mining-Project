
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
nltk.download('wordnet')
nltk.download('omw-1.4')


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
        lem_word = lemmatizer.lemmatize(word)
        #lem_word.append(stem_word)
        lem_word = lem_word + " " + word
        
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

#n=len(tweet_data)
#n_train = int(n*0.80)
#n_test = int(n*0.20)

#ample_train = tweet_data[:n_train]
#sample_test = tweet_data[n-n_test:]


#label_train = tweet_label[0:n_train]
#label_test = tweet_label[n-n_test:]

#Create frequency table based on sample data
#freqs=frequency_builder(sample_train,label_train)


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

freqs=frequency_builder(train_sample,train_y)
print("Frequency Build done")

with open('saved_frequency.pkl', 'wb') as f:
    pickle.dump(freqs, f)

n,p=np.array(train_y).shape
print("Training Labels:",n,p)

n,p=np.array(test_y).shape
print("Testing Labels",n,p)


X = np.zeros((len(train_sample), 3))
for i in range(len(train_sample)):
    X[i, :]= extract_features(train_sample[i], freqs)

Y = train_y
B=np.zeros((3, 1))
alpha=0.01
B=gradient_Descent(X, Y, B, alpha, 1500)

print(B)
Label_pred = []
for tweet in test_sample:
    y_pred = predict_tweet(tweet, freqs, B)[0][0]
    #print(y_pred)
    if y_pred > 0.40:
        Label_pred.append(4)
    else:
        Label_pred.append(0)


Label_pred = np.array(Label_pred)
test_y = test_y.reshape(-1)
accuracy = np.sum((test_y == Label_pred).astype(int))/len(test_sample)
print(label_test[0:100])
print("----------------")
print(Label_pred[0:100])
mse_test_N=mean_squared_error(label_test,Label_pred)
print(accuracy)
print(mse_test_N)