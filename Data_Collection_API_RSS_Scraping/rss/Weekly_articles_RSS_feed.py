'''
Take the weekly podcast or articles from Naval, Farnam Street, and Seth Godin's blog.
'''

import feedparser
import os
import pandas as pd

from datetime import datetime


# set the feeds as global variables
Naval = "https://nav.al/podcast/feed"
Farnam_street = "https://fs.blog/feed/"
Seth = "http://sethgodin.typepad.com/seths_blog/index.rdf"

RSS_List = [Naval, Farnam_street, Seth]

#set the dataframe columns as global variables
df_columns = ['source', 'date', 'rank', 'title', 'link', 'summary']

#create a function to parse data and append to a dataframe
def top10_articles(rss, df):

    #parse the feed
    feed = feedparser.parse(rss)

    #check bozo to see if our feed is well formed
    if feed.bozo == 0:
        print("%s is a well-formed feed!" % feed.feed.title)
    else:
        print("%s has flipped the bozo bit. Potential errors ahead!" % feed.feed.title)

    #set the feed date to be the published date, if it exists. If not, use the current date
    feed_date = feed.feed.get('published', datetime.now().strftime('%Y-%m-%d'))

    #set a counter for our loop
    i = 0

    #for the first 10 movies in our feed, append the required information to the dataframe
    while i < 10:
        feed_items = pd.Series([feed.feed.title, feed_date, i+1, feed.entries[i].title, feed.entries[i].id, \
                       feed.entries[i].summary], df_columns)
        df = df.append(feed_items, ignore_index = True)
        i+= 1

    #return the dataframe
    return df

if __name__ == "__main__":

    #create an empty dataframe
    top10_df = pd.DataFrame(columns = df_columns)


    for rss in RSS_List:
    	top10_df = top10_articles(rss, top10_df)

    #save the dataframe to a csv. if the csv exists, append to it
    if not os.path.isfile('weekly_articles.csv'):
    	top10_df.to_csv('weekly_articles.csv', header = df_columns, index=False)
    else:
    	top10_df.to_csv('weekly_articles.csv', mode = 'a', header=False, index=False)
