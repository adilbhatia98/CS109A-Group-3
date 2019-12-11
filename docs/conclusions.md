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
Our best model is the boosted decision tree classifier with a depth of 2 and 751 iterations, 
which performs with an accuracy of 95.4% in the training set and 93.0% in the test set.

The following table summarizes the accuracies for all our models, ordered by accuracy in the test set:
```
feature	        importance
minutes_y	        0.149895
30_day_perc	        0.126930
14_day_perc	        0.097110
Last Price	        0.094082
24_hr_perc	        0.087268
7_day_perc	        0.071768
tweet_len	        0.071687
sent_score	        0.059620
month	            0.057562
hour	            0.04439
```

<p>These feature importances on our best model (ADA Boosting with time interval 60 minutes) demonstrates counterintuitively that the most important feature to predict the movement of VIX data post a trump tweet is what time he tweeted. The important thing to note is that VIX trades global hours, so the trading activity begins 3:15am and ends 4:15pm (with a 15 minute break between 915am and 930am). This suggests two possible factors for the importance of the time Trump tweets. Firstly, if the time Trump tweets is non-US market hour times, then the immediate impact (in the 60 minute time interval) is less profound because while there will be some traders looking to trade based on his tweets even outside US Market Hours, it is likely that most of the traders that closely trade on Trump’s actions are based in US and that most of them are trading during Market hours (the argument is not that there are not traders in China/other places trading off Trump’s tweets/actions, but that the proportion of them is much higher in US which ultimately is reflected more in the VIX). Secondly, on similar lines, this also would be further demonstrated if we had access to Trading Volume minute by minute. Some hours are more likely to have higher trading volume than other times, so the opportunity for a Trump tweet to cause significant impact on Volatility would also be determined by the size of the trading volume at that given time. However, this was incredibly challenging to collect because VIX does not have trading volume data itself, but rather we would have had to collect minute by minute volumes from the put and call option derivates on the S&P, which was not compiling cleanly with the VIX minute by minute data that we had collected from the Bloomberg Terminals at HBS Baker Library.</p>
<p>Additionally, the next three predictors all have to do with the momentum of the VIX prices, rather than the tweets themselves. This suggests two potential things. Firstly, VIX data is influenced by combination of things and even if we look so closely just sixty minutes after the tweet, there are still market forces that are a continuing trend that then combine with the Trump tweet effect to influence prices on VIX.</p>
<p>Tweet Length is incredibly interesting, as it probably indicates that a longer tweet likely (though not always) suggests that this tweet is more substantive with bigger implications thus causing further movement on the trading activity. We then continue with further VIX data points with momentum related predictors and time (now, in the form of what month it was tweeted). Although not as much as would have thought initially, but sentiment score has some quantifiable significance to the VIX Data. We predict that the reason for this is because his positive tweets are not necessarily market reassuring to lead to VIX data decreasing nor are his negative tweets necessarily acted upon by the market even if they have content that is related to the market, as he has in the past often threatened no deal which would be classified as a negative sentiment, but the market may ignore this and other factors impacting the VIX at that time, making sentimental scores not as powerful as initially predicted. Moreover, in hindsight, we think a way to control for this is to create interaction terms with keywords such as ‘China’ and ‘Deal’.  On the topic of keywords, we were surprised for a second that any of our keyword related predictors were not present in our top features. We realized that the keywords were separate one-hot-encoded predictors such that any keyword itself did not have enough data-points to consistently predict the VIX Data, but this in hindsight would have been really helpful to create multiple interaction terms among several different keywords and all the keywords to observe if they are then found more appealing by the model.</p>

# Limitations and Future Work
### Data Size
{: .no_toc }
<p>We generated a dataset by consolidating and cleaning the Trump's tweets since he became the presumptive GOP candidate in 2016. It would be interesting to expand our data to before he was President/in the running and see how the effects of his Tweets changed if all. We could also rn variations of our analysis using keyword-based subset of the Twitter data to see if there is interesting sentiment analysis in these specific Tweets Trumps posts.</p>

### Improve Neural Network
{: .no_toc }
<p>Idealy, we would further optimize the Neural Network model as its accuracy is currently behind ADA Boosting by about 6-7% points. We tried several different layers, regularization techniques, optimizers, batch-sizes, etc. Unfortunately, we kept leveling around the 60-62% range in our accuracy. We also adapted the data we were training the model for, as the target outcome no longer had three categories, but instead only two (just positive and negative as our threshold previously was already very low at 0.01), as our binary_cross_entropy was working better than the respective loss function for the multi-classification model (in the sense that latter was outputting ridiculously low accuracy scores).</p>

### Sentiment Analysis Improvement
{: .no_toc }
<p>Finally, a review of the literature highlighted the importance of clarifying and classifying sentiment more sepcifically. We could improve this model by loading specific emotion datasets to improve our sentiment analysis and take scoring beyond just positive and negative via textblob's fundamental functionality. Utilizing these libraries could help make our sentiment analysis more robust. This would be particularly important given sentiment score was not the most important predictor in our model.</p>


