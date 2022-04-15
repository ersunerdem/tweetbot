#Modified code from https://medium.com/dataseries/how-to-scrape-millions-of-tweets-using-snscrape-195ee3594721
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import sys
import codecs

#Constants
num_tweets = 500    #Number of tweets to scrape for data
num_test = 50  #Number of num_tweets designated to go to test_data
#File name for the training data
training_fn = "training_data.txt"
#File name for test data
test_fn = "test_data.txt"
since_data_str = '2016-01-01'
users=['KurgerBang', 'dril', 'internethippo', 'danmentos', 'A_Person547']
search_query = f'since:{since_data_str} lang:en exclude:replies exclude:retweets'


if(num_test >= num_tweets):
    print('ERROR: num training must be smaller than num_tweets!')

tweets_list = []

for n, u in enumerate(users):
    mod_query = f'from:{u} ' + search_query
    print(f'Scraping Twitter for tweets matching query: {mod_query}')

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(mod_query).get_items()):

        if i>=num_tweets:
            print(f'Finished grabbing {num_tweets} tweets.')
            break
        elif i%500==0 and i > 0:
            print(f'Grabbed {i} tweets so far.')
    
        tweets_list.append([tweet.content])

df = pd.DataFrame(tweets_list, columns=['tweet'])


#Here's where the tutorial ends... save data to test and training data

print('Number of actual tweets grabbed:', len(df['tweet']))

#Separate into training and test data
training_data = df['tweet'][0:(len(df['tweet'])-num_test)]
test_data = df['tweet'][num_test:(len(df['tweet'])-1)]

#Write to files
print(f'Writing to {training_fn}...')
with open(training_fn, 'a', encoding='utf-8') as f:
    dfAsString = training_data.to_string(header=False, index=False)
    f.write(dfAsString)

print(f'Writing to {test_fn}...')
with open(test_fn, 'a', encoding='utf-8') as f:
    dfAsString = test_data.to_string(header=False, index=False)
    f.write(dfAsString)