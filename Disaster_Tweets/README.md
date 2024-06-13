# Disaster Tweets

Data: https://www.kaggle.com/competitions/nlp-getting-started/data

Dataset of tweets in which the task is to determine if the tweet is from an actual disaster or not. There is a variety of different disasters included and also a variety of tweets that utilize words metaphorically to describe things which to a human would be easy to understand, but a machine may see things otherwise.

My approach was to utilize word-embeddings to get a vector representation of the tweets. Then I passed these embeddings through a network to predict if it was an actual diaster or not. With this meteoric rise in LLM's there are so many different embeddings out now I'm sure any of them would work decent.

The disaster_tweets.py file contains a straightforward solution with most of the extra stuff stripped out. However, the NLP_Natural_Disaster notebook contains a bit of data exploration as well as a good amount of model diagnostics and tests to get an idea of how effective the training runs were going. 

I've been implementing the general approach of Karpathy's 'Recipe for Training Neural Networks' in an effort to improve my overall efficiency when solving these problems. So often the workflow when working on deep learning projects is all over the place and I want to be better about trying to avoid that and have a more steady workflow that's applicable to any future job/freelance situations.

I also developed a web application for this code. If you run the python file locally to obtain the trained model you can interact with the model through the web application and test the results you get for different tweets that you enter!

![web app screenshot](https://github.com/Stepheni12/Deep_Learning_Projects/assets/26886046/51719188-e41c-4c9b-a862-e3c3b828edf2)
