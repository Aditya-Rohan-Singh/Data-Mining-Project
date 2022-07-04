import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import re
import string
import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

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

    #new_tweet = re.sub('.','', new_tweet)

    # instantiate tokenizer class
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    
    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(new_tweet)    
        
    #Import the english stop words list from NLTK
    stopwords_english = stopwords.words('english') 
    stopwords1 = set(STOPWORDS)
    stopwords1.update(["br", "href","https","t","co","c","b'RT","b'","'","b","rt"])
    
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
    return lem_word

def predict_tweet(tweet, freqs, B):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid_func(x,B)
    return y_pred

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

def frequency_builder(sample_train, label_train):
    label_train_list = np.squeeze(label_train).tolist()
    
    freqs = {}
    for y, tweet in zip(label_train_list, sample_train):
        for word in process_tweet(tweet):
                pair = (word, y)
                freqs[pair] = freqs.get(pair, 0) + 1
    return(freqs)

#Loading frequency dictionary of training data
with open('saved_frequency.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

#Loading the Beta matrix from Logisitc Regression
with open('Beta.pkl', 'rb') as f:
    loaded_dict_B = pickle.load(f)

encoding_used = "ISO-8859-1"
test_sample = pd.read_csv("Tweets.csv", header = None, encoding = encoding_used)
test_sample.columns = ["tweet"]
sample = test_sample


test_sample = np.array(test_sample)

#Cleaning data
test_sample1 = []
for i in range(len(test_sample)):
    test_sample1.append(test_sample[i][0])


Label_pred = []
#Predicting data from B value
for tweet in test_sample:
    tweet = tweet[0]
    y_pred = predict_tweet(tweet, loaded_dict, loaded_dict_B)[0][0]
    #print(y_pred)
    if y_pred > 0.45:
        Label_pred.append(1)
    else:
        Label_pred.append(0)


#Saving CSV files tweet and its predicted label
Label_pred = np.array(Label_pred)
sample["label"] = Label_pred
prediction_freq = frequency_builder(test_sample1, Label_pred)
positive = sample[sample['label'] == 1]
negative = sample[sample['label'] == 0]
sample.to_csv('predicted_data.csv')
positive.to_csv('positive.csv')
negative.to_csv('negative.csv')



#Storing the top 10 positive words and negative words
words = sorted(prediction_freq, key=prediction_freq.get, reverse=True)#[:10]

i = 0
j = 0
positive_words = []
negative_words = []
for x, y in words:
    if(y == 1):
        positive_words.append(x)
    elif(y==0):
        negative_words.append(x)
       
positive_words = positive_words[:10]
negative_words = negative_words[:10]
top_words = pd.DataFrame(positive_words,columns=["Positive"])
top_words["Negative"] = negative_words
top_words.to_csv("top_words.csv")
print(top_words)


#Displaying and saving bargraph
result = sample.groupby(['label']).count()
result.plot(kind="bar")
xlabel = ["Positive(1)", "Negative(0)"]
pos = mpatches.Patch(color='blue', label='Positive(1)')
neg = mpatches.Patch(color='blue', label='Negative(0)')
plt.legend(handles=[pos,neg], loc=2)
plt.savefig('bar_graph.jpeg')
#plt.show()