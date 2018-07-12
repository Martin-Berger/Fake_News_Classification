from pprint import pprint
import requests
import json
from pandas import *
import praw
import pprint
from collections import Counter, defaultdict
import numpy as np
import multiprocessing as multi
import random
import math
import tldextract
from time import sleep
from tqdm import trange


def init_Reddit():
    return(praw.Reddit(client_id='M0oItMPvV6eScQ',
                     client_secret='WBrM6FF3zpD_Ok6dBth2FK5x8ws',
                     user_agent='Script:fake.news.predictor:1.0 (by /u/gargomeister)',
                     password='USEYOUROWNPASSWORD', username='gargomeister'))


def chunks(n, source_list):
    """Splits the list into n chunks"""
    return np.array_split(source_list,n)

def launch_multi_process(function, source_list, r):
    manager = multi.Manager()
    result = manager.list()
    cpus = multi.cpu_count()
    workers = []
    source_bins = chunks(cpus, source_list)
    for cpu in range(cpus):
        worker = multi.Process(name=str(cpu), 
                               target=function, 
                               args=(r, source_bins[cpu].tolist(),result,))
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()
    return result



def perform_training_search_extraction(r, page_ranges, res):
    for i in trange(len(page_ranges), desc='Domain Scraping'):
        try:
            page_response = r.get("search", {"q": page_ranges[i], "limit": "25", "sort": "relevance"})
            for j in trange(len(page_response), desc='Posts Scraping', leave=False):
                result = page_response[j]
                try:
                    current_user = r.redditor(result.author.name)
                    subreddit = r.subreddit(result.subreddit_name_prefixed[2:])
                    comments = [comment.body for comment in result.comments.list()[1:10]]
                    res.append([result.id, page_ranges[i], result.author.name, result.title, result.num_comments, result.subreddit_subscribers, result.subreddit_name_prefixed, result.selftext, comments, result.ups, current_user.created, current_user.has_verified_email, current_user.is_gold, current_user.is_mod, current_user.link_karma, current_user.comment_karma, subreddit.active_user_count, subreddit.advertiser_category, subreddit.audience_target])
                except:
                    pass
                sleep(0.01)
        except:
            pass


def create_df_from_posts(posts):
    transpose = list(zip(*posts))
    raw_data = {
        "post_id": transpose[0], 
        "website": transpose[1], 
        "post_author": transpose[2], 
        "post_title": transpose[3],
        "post_num_comments": transpose[4],
        "post_subreddit_subscribers": transpose[5],
        "post_subreddit_name_prefixed": transpose[6],
        "post_selftext": transpose[7],
        "comments": transpose[8],
        "post_upvotes": transpose[9],
        "user_created": transpose[10],
        "user_has_verified_email": transpose[11],
        "user_is_gold": transpose[12],
        "user_is_mod": transpose[13],
        "user_link_karma": transpose[14],
        "user_comment_karma": transpose[15],
        "subreddit_active_user_count": transpose[16],
        "subreddit_advertiser_category": transpose[17],
        "subreddit_audience_target": transpose[18]
    }
    df = pandas.DataFrame(data=raw_data, columns=["post_id", "website", "post_author", "post_title", "post_num_comments", "post_subreddit_subscribers", "post_subreddit_name_prefixed", "post_selftext", "comments", "post_upvotes", "user_created", "user_has_verified_email", "user_is_gold", "user_is_mod", "user_link_karma", "user_comment_karma", "subreddit_active_user_count", "subreddit_advertiser_category", "subreddit_audience_target"])
    return df

def main():

    r = init_Reddit()
    sources = read_csv('sources_clean.csv')
    source_list = np.array(sources["site"])
    posts = launch_multi_process(perform_training_search_extraction, source_list, r)
    print(len(posts))
    df = create_df_from_posts(posts)
    #Append label to the training data-set
    df["doubt"] = 0 
    df["fake"] = 0
    df["reliable"] = 0
    df.info()

    for index, observation in sources.iterrows():
        if observation["satire"] == 1 or observation["bias"] == 1:
            df.loc[df.website == observation.site, ["doubt"]] = 1
        elif observation["reliable"] == 1:
            df.loc[df.website == observation.site, ["reliable"]] = 1
        else:
            df.loc[df.website == observation.site, ["fake"]] = 1   

    df.to_csv('post_training.csv', index=False)



if __name__ == '__main__':
    main()