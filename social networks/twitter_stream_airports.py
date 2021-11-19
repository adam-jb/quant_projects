

# works for tweepy 3.9 but not 3.3 or 4.1
# sudo pip install -Iv tweepy==3.9.0

# think twitter allows 900 tweets to be streamed every 15 mins:
# https://developer.twitter.com/en/docs/twitter-api/rate-limits

# You can retrieve the last 3,200 tweets from a user timeline and search the last 7-9 days of tweets
# source: https://gwu-libraries.github.io/sfm-ui/posts/2017-09-14-twitter-data


# options for historic twitter:
# https://developer.twitter.com/en/products/twitter-api/enterprise
# possible free trial: https://keyhole.co/





from tweepy import OAuthHandler, API, Cursor
from config import *
import datetime
import pandas as pd
import os
import time
import configparser



# get tokens from other file
exec(open('/Users/apple/Desktop/twitter_api_keys.py').read())



auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth,wait_on_rate_limit=True)


# setting initial dates
start_day = datetime.date.today()
previous_day = start_day - datetime.timedelta(days=1)


print('starting')
#time.sleep(751)  # waiting 15 mins to ensure all tweets captured for that day


# looping to get tweets over a year
for i in range(355):

	tweets_list = Cursor(api.search, q="@HeathrowAirport OR @STN_Airport OR @Gatwick_Airport OR @manairport OR @LDNLutonAirport since:" + str(previous_day)+ " until:" + str(start_day),tweet_mode='extended', lang='en').items()

	output = []
	for tweet in tweets_list:
	    text = tweet._json["full_text"]
	    favourite_count = tweet.favorite_count
	    retweet_count = tweet.retweet_count
	    created_at = tweet.created_at
	    print(text)

	    # store image URLs
	    media_urls = []
	    if 'media' in tweet.entities:
	    	for media in tweet.extended_entities['media']:
	    		media_urls.append(media['media_url'])
	    		print(media['media_url'])


	    # storing data
	    line = {'text' : text, 'favourite_count' : favourite_count, 'retweet_count' : retweet_count, 'created_at' : created_at, 'media_urls': media_urls}

	    output.append(line)

	output = pd.DataFrame(output)

	print('saving output for ' + str(previous_day) + ' with shape of ' + str(output.shape))
	output.to_csv('/Users/apple/Desktop/quant_projects/social networks/top_5_airport_files/a' + str(previous_day) + '.csv')

	# updating query dates
	start_day = start_day - datetime.timedelta(days=1)
	previous_day = previous_day - datetime.timedelta(days=1)

	time.sleep(751)  # waiting 15 mins to ensure all tweets captured for that day





