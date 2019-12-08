---
title: Background
layout: default
nav_order: 2
---

# Background
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
# Project Question

How do President Trump’s tweets impact the volatility of the S&P 500? How has that impact changed over time, if at all, since he began his term as U.S. President?

# Project Motivation

In September 2019, J.P. Morgan launched their 'Volfefe' index meant to track U.S. treasury bond volatility based on market sentiment changes due to unpredictable tweets from President Trump. Using a portion of Trump's tweets since 2017, JPM developed a model that predicted bond market volatility surprisingly accurately.

In our project, we examine VIX, the first benchmark index to measure expectations of future market volatility. VIX prices are based on S&P 500 options, and rapid and/or significant price changes usually reflect sudden changes in market sentiment. For example, if VIX jumps 10% in a given day, it suggests that investors are generally more unsure about the future of the U.S. economy, resulting in greater volatility in the overall market.

We had three goals with this project:
1. create a well-performing model that can predict changes in VIX based on Trump's tweets
2. use that model to generate positive returns in the coming future
3. produce a clean interface using <i>jekyll</i> to display our project analysis and results so that others can replicate our method and hopefully improve it!

# Approach
UPDATE
We first downloaded Trump's twitter database and cleaned the data to focus on the tweet content. Then, we analyzed it using <i>textblob<i> to generate a sentiment score for each tweet. Next, we went to HBS and manually pulled minute-by-minute VIX data (a cumbersome process, but one we believe was worth the effort!). Then, we consolidated this data into a single dataframe.
We then randomly split this data into a training and test set with the outcome variable being the change in VIX pricing and the core explanatory variable being the sentiment score for a given tweet, along with several other predictors constructed based on the twitter and VIX data. Predicting the absolute VIX price yielded irrelevant results (nonsurprisingly), but predicting the change in VIX price produced some interesting results.
	
Next, we built and fit a variety of classifiers with differing VIX price change time intervals. Models we built include:
- Baseline Model
- Logistic Classifier

For each model, we evaluated its accuracy on both our training and and test set. 
Based on accuracy scores, we determined the classifier with the highest performance on the test set. 
Finally, we ran our best-performing model on a fresh set of songs and asked Grace if she liked her new playlist.

---


# Literature Review 
[Using Twitter to Predict the Stock Market](https://link-springer-com.ezp-prod1.hul.harvard.edu/content/pdf/10.1007/s12599-015-0390-4.pdf)
+ Attemt to extract mood levels from Social Media applications in order to predict stock returns
+ Uses roughly 100 million tweets that were published in Germany between January, 2011 and November, 2013
+ Find that it is necessary to take into account the spread of mood states among Internet users
+ Portfolio increases by up to 36 % within a sixmonth period after the consideration of transaction costs

[Data Mining Twitter To Predict Stock Market Movements](https://doaj.org/article/9b96accf4c704685ab634ade80b10978)
+ Sentiment analysis of Twitter data from July through December, 2013 
+ Attempt to find correlation between users’ sentiments and NASDAQ closing price and trading volume
+ Find that “Happy” and “sad” sentiment variables’ lags are strongly correlated with closing price and “excited” and “calm” lags are strongly correlated with trading volume
+ Incorporates interesting word weighting to classify tweets into emotional groupings

[A topic-based sentiment analysis model to predict stock market price movement using Weibo mood](http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/pdfviewer/pdfviewer?vid=1&sid=7f8ab0dd-7895-42af-b230-9572b6558b2a%40sdc-v-sessmgr02)
+ Demonstrate that the emotions automatically extracted from the large scale Weibo posts represent the real public opinions about some special topics of the stock market in China.
+ Nonlinear autoregressive model, using neural network with exogenous sentiment inputs is proposed to predict the stock price movement.
+ "Given the performance, if more related topics for a stock are found out, the accuracy could be higher considering more completed topic-based sentiment inputs."
+ Recognize that the time lag on sentiment makes it difficult to fully predict market changes

[Can Twitter Help Predict Firm-Level Earnings and Stock
Returns?](http://web.a.ebscohost.com.ezp-prod1.hul.harvard.edu/ehost/pdfviewer/pdfviewer?vid=1&sid=ec9a1b56-07fd-4d64-96ed-ff2f541510f8%40sessionmgr4007)
+ Test whether opinions of individuals tweeted just prior to a firm’s earnings announcement predict its earnings and announcement returns
+ Find that the aggregate opinion from individual tweets successfully predicts a firm’s forthcoming quarterly earnings and announcement returns

