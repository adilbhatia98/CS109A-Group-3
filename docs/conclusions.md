---
title: Conclusions
layout: default
nav_order: 5
---

# Conclusions
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Analysis of Best Model
Our best model is the boosted decision tree classifier with a depth of 10 and 5 iterations, 
which performs with an accuracy of 66.8% in the test set.

The following table summarizes the top 10 features in our best model:
```
feature	        importance
minutes_y	        0.149895
30_day_perc	      0.126930
14_day_perc	      0.097110
Last Price	      0.094082
24_hr_perc	      0.087268
7_day_perc	      0.071768
tweet_len	        0.071687
sent_score	      0.059620
month	            0.057562
hour	            0.04439
```

These feature importances on our best model (ADA Boosting with time interval 60 minutes) demonstrates counterintuitively that the most important feature to predict the movement of VIX data post a trump tweet is what time he tweeted. The important thing to note is that VIX trades global hours, so the trading activity begins 3:15am and ends 4:15pm (with a 15 minute break between 915am and 930am). This suggests two possible factors for the importance of the time Trump tweets. Firstly, if the time Trump tweets is non-US market hour times, then the immediate impact (in the 60 minute time interval) is less profound because while there will be some traders looking to trade based on his tweets even outside US Market Hours, it is likely that most of the traders that closely trade on Trump’s actions are based in US and that most of them are trading during Market hours (the argument is not that there are not traders in China/other places trading off Trump’s tweets/actions, but that the proportion of them is much higher in US which ultimately is reflected more in the VIX). Secondly, on similar lines, this also would be further demonstrated if we had access to Trading Volume minute by minute. Some hours are more likely to have higher trading volume than other times, so the opportunity for a Trump tweet to cause significant impact on Volatility would also be determined by the size of the trading volume at that given time. However, this was incredibly challenging to collect because VIX does not have trading volume data itself, but rather we would have had to collect minute by minute volumes from the put and call option derivates on the S&P, which was not compiling cleanly with the VIX minute by minute data that we had collected from the Bloomberg Terminals at HBS Baker Library. Also, note that the way our minutes variable is constructed technically precludes the need for an hours variable, something that we could have changed in order to make our model 'cleaner.'

Additionally, the next three predictors all have to do with the momentum of the VIX prices, rather than the tweets themselves. This suggests two potential things. Firstly, VIX data is influenced by combination of things and even if we look so closely just sixty minutes after the tweet, there are still market forces that are a continuing trend that then combine with the Trump tweet effect to influence prices on VIX.

Tweet Length is incredibly interesting, as it probably indicates that a longer tweet likely (though not always) suggests that this tweet is more substantive with bigger implications thus causing further movement on the trading activity. We then continue with further VIX data points with momentum related predictors and time (now, in the form of what month it was tweeted). Although not as much as would have thought initially, but sentiment score has some quantifiable significance to the VIX Data. We predict that the reason for this is because his positive tweets are not necessarily market reassuring to lead to VIX data decreasing nor are his negative tweets necessarily acted upon by the market even if they have content that is related to the market, as he has in the past often threatened no deal which would be classified as a negative sentiment, but the market may ignore this and other factors impacting the VIX at that time, making sentimental scores not as powerful as initially predicted. Moreover, in hindsight, we think a way to control for this is to create interaction terms with keywords such as ‘China’ and ‘Deal’.  On the topic of keywords, we were surprised for a second that any of our keyword related predictors were not present in our top features. We realized that the keywords were separate one-hot-encoded predictors such that any keyword itself did not have enough data-points to consistently predict the VIX Data, but this in hindsight would have been really helpful to create multiple interaction terms among several different keywords and all the keywords to observe if they are then found more appealing by the model.

# Limitations and Future Work
### Data Size
{: .no_toc }
We generated a dataset by consolidating and cleaning the Trump's tweets since he became the presumptive GOP candidate in 2016. It would be interesting to expand our data to before he was President/in the running and see how the effects of his Tweets changed if all. We could also rn variations of our analysis using keyword-based subset of the Twitter data to see if there is interesting sentiment analysis in these specific Tweets Trumps posts.

### Improve Neural Network
{: .no_toc }
Ideally, we would further optimize the Neural Network model as its accuracy is currently behind ADA Boosting by about 6-7% points. We tried several different layers, regularization techniques, optimizers, batch-sizes, etc. Unfortunately, we kept leveling around the 60-62% range in our accuracy. We also adapted the data we were training the model for, as the target outcome no longer had three categories, but instead only two (just positive and negative as our threshold previously was already very low at 0.01), as our binary_cross_entropy was working better than the respective loss function for the multi-classification model (in the sense that latter was outputting ridiculously low accuracy scores).

### International Markets
{: .no_toc }
We would also have benefitted from incorporating price and trading data from other markets—most notably, the Chinese stock market. Trump’s tweets, particularly tweets related to the trade war with China, are likely followed closely by traders there and would have a market-moving effect. Further, adding Chinese trading hours to our dataset would have allowed us to broaden the range of tweets that could be incorporated into our predictive model. It is also possible that the impact of trade-related tweets would be easier to isolate with data on the Chinese market, because the subset of Trump tweets that plausibly move stock prices in China would be narrower than those that have an impact on American shares. This reduces the possibility that our China trade tweet signals would be confounded by tweets containing those signals that are not related to the Chinese trade issue. Also, the actual market movements would be sharper, insofar as evidence suggests that the current trade dispute has a more negative impact on Chinese equities than it does on American equities.

### Sentiment Analysis Improvement
{: .no_toc }
Finally, a review of the literature highlighted the importance of clarifying and classifying sentiment more sepcifically. We could improve this model by loading specific emotion datasets to improve our sentiment analysis and take scoring beyond just positive and negative via textblob's fundamental functionality. Utilizing these libraries could help make our sentiment analysis more robust. This would be particularly important given sentiment score was not the most important predictor in our model.

### Regression Functional Form
{: .no_toc }
As briefly mentioned above, we could improve the interpretation and usefulness of the model by including interaction terms, particularly incorporating some of the indicator variables. Additionally, we could try to include quadratic terms to see how that changes the effects of certain variables in our analysis.

# Final Thoughts
Overall, we were proud that we were able to improve our model such that test accuracy was not only above 50%, but also that it reached nearly 70%. Additionally, the fact that our model successfully reached this accuracy level for our 60 min interval offers a great trading opportunity based on the VIX index. Based on the above comments, though, we do believe more tweaking should be done in order to improve the model and make it more robust if it were to be using in an actual trading context.
