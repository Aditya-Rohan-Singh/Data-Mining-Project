import tweepy as tw
import csv

consumer_key= '5limczXPkrQXwSKZBgr22xvRY'
consumer_secret= 'dnxoBGO2F0DptXDu6rQfBFf6E9tqWDFSh5SYvbh1mNAjKb8QhG'
access_token= '110668153-9xkWmU2L83wl6EyrYAocda9ITYsXeb7BThzbbDiv'
access_token_secret= '2NgrIpmwUjJU33FDhNtSs0WZNmnxs3Hjv4fbCfmZfNQZR'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

search_words = "Depp"

#date_since = "2021-10-01"

tweets = tw.Cursor(api.search_tweets, q=search_words, lang="en").items(2000)

df = open('Tweets.csv', 'w', encoding="utf-8", newline="")

write=csv.writer(df)

for tweet in tweets:
    write.writerow({tweet.text.encode('utf-8')})  

df.close()