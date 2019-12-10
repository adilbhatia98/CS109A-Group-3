---
title: Data Exploration
layout: default
nav_order: 3
---
# Data Exploration
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

<!-- 
<style>
blockquote { background: #AEDE94; }
h1 { 
    padding-top: 25px;
    padding-bottom: 25px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
h2 { 
    padding-top: 10px;
    padding-bottom: 10px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}

div.exercise {
	background-color: #ffcccc;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
}
div.theme {
	background-color: #DDDDDD;
	border-color: #E9967A; 	
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 18pt;
}
div.gc { 
	background-color: #AEDE94;
	border-color: #E9967A; 	 
	border-left: 5px solid #800080; 
	padding: 0.5em;
	font-size: 12pt;
}
p.q1 { 
    padding-top: 5px;
    padding-bottom: 5px;
    text-align: left; 
    padding-left: 5px;
    background-color: #EEEEEE; 
    color: black;
}
header {
   padding-top: 35px;
    padding-bottom: 35px;
    text-align: left; 
    padding-left: 10px;
    background-color: #DDDDDD; 
    color: black;
}
</style>
 -->


# Data Collection and Cleaning


## Twitter
We pulled all of Trump's tweets in the last few years from his Twitter archive. Due to the naature of the database, we spent much time cleaning this data for our purposes. We took the following steps to clean the data:
+ Importing raw data from the archive
+ Removing unnecessary columns
+ Adjusting GMT to Eastern Time + accounting for daylight savings
+ Manually fixing errors in cells where the delimiting was incorrectly done in the database output and manually re-inserting the delimiting character
	
<p>First, we removed all Twitter data preceding June 1, 2016. Trump became the presumptive nominee of the Republican Party that summer, and we figured that this represented the beginning of Trump tweets’ meaningful market relevance. 
	There remained numerous instances of tweet data from the database lumping together multiple tweets in a single entry. We noticed that when this occurred, the components of these tweets were delimited by the character “{“. We used Excel’s “find” function to filter for all such cases. We then transferred all these “mega-cells” into a separate Excel sheet, which automatically separated the subcomponents of the tweet data: each row represented the data, including the actual tweet string, for a given tweet. However, the various classes of data (e.g., text, id_str, date, etc.) were lumped together in one column. These were split  using the aforementioned delimiter. We appended these cleaned data into the original Excel sheet, and then sorted the entire sheet by date to complete the process.
	Additionally, tweets that were subject to the above issue—and many others, in general—also misclassified retweets as Trump’s original tweets. Retweets started with ‘@[handle] : ‘ so I wrote a function to filter through the data for entries starting with ‘@‘ that also included a ‘:’. Not all such instances were retweets—some were just tweets by Trump—and so this process had a manual component too.</p>
	
We utilized `ntlk`'s `textblob` function in order to analyze the sentiment of tweets in our data set. For each tweet, this function created a polarity score (the more positive a tweet is, the closer the score is to 1; the more negative, the closer it is to -1). The function also returns a subjectivity score. Lower subjectivity score means that the tweet's polarity score more objectively represents its sentiment.

## VIX
VIX is the first benchmark index to measure expectations of future market volatility 
	(based on S&P 500 options). Since Milestone 2, we have procured minute-by-minute VIX data from 12/2015 - 11/11/2019. 
	This data includes only VIX pricing on trading days throughout the past few years (excludes holidays, weekends, etc.). 
	We manually pulled the data from a Bloomberg Terminal at HBS Baker Library. Given the size of the dataset and the Terminal download limits, we manually copied and pasted the VIX data directly into a csv file. Note that all other sources of VIX data are at best day-by-day and typically cost a nontrivial amount.

VIX is managed by CBOE (Chicago Board Options Exchange). The global trading hours for VIX can be found [here]("https://www.cboe.com/micro/eth/pdf/global-trading-hours.pdf"). Trading hours range from 3:15 am EST until 4:15 pm EST. There is a break bewteen 9:15 and 9:30 am, but this is addressed in how we consolidate our data.

## Consolidated Data
In order to consolidate the data, we merge our Twitter and VIX dataframes on date/time to ensure that we are only looking
	at tweets that are posted during VIX trading hours. That way, we do not have to worry about tweets occurring outside
	these hours. Also, when we look at the change in VIX price, we are only looking at changes during trading hours, so
	examining only tweets that are posted during trading hours allows us to perform this analysis soundly.

```python
from textblob import TextBlob
tweets = twitter_archive_df['text']
tweets_list = [tweet for tweet in tweets]
big_tweet_string = ' '.join(tweets_list)

tokens = word_tokenize(big_tweet_string)
words = [word.lower() for word in tokens if word.isalpha()]

stop_words = set(stopwords.words('english'))
words = [word for word in words if not word in stop_words]

scores = []
subjectivity = []
for tweet in tweets_list:
    blob_tweet = TextBlob(tweet)
    sentiment = blob_tweet.sentiment
    score = sentiment[0]
    subject = sentiment[1]
    
    scores.append(float(score))
    subjectivity.append(float(subject))

twitter_archive_df['sent_score'] = scores
twitter_archive_df['subjectivity'] = subjectivity
# twitter_archive_df.head()
```


<hr style="height:2pt">

# Data Description
Our data includes the following features:
- `danceability`: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
- `energy`: Energy represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. A value of 0.0 is least energetic and 1.0 is most energetic. 
- `key`: The estimated overall key of the track. Integers map to pitches using standard Pitch Class Notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- `loudness`: The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values range between -60 and 0 db. 
- `mode`: Mode represents the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Mode is binary; major is represented by 1 and minor is 0.
- `speechiness`: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- `acousticness`: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. 
- `instrumentalness`: Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- `liveness`: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. 
- `valence`: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- `tempo`: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 
- `duration_ms`: The duration of the track in milliseconds.
- `time_signature`: An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
- `popularity`: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are.Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. 
- `in_playlist`: Response variable. Categorical variable for whether in playlist of desire. 1 if in playlist, 0 if not in playlist.

The following features were recorded to help with visualization later, but not used as predictors in our analysis, as they are not characteristics of the music itself.
- `name`: Song title
- `artist`: First artist of song
- `type`: The object type, always deemed 'audio_features.'
- `id`: The Spotify ID for the track.
- `uri`: The Spotify URI for the track.
- `track_href`: A link to the Web API endpoint providing full details of the track.
- `analysis_url`: An HTTP URL to access the full audio analysis of this track. An access token is required to access this data.

# Exploratory Data Analysis

We did an initial analysis of some keywords, as shown in our notebook. The list is `['trade', 'Trade', 'deal', 'products', 'manufacturing', 'China', 'Xi', 'Xi Jinping', 'CCP', 'Communist Party', 'Beijing']`. Approximately 1/10 of Trump’s tweets that we cleaned (of ~14,000 total) contain some combination of these keywords. As we see above, the majority of the 'keywords' (based on our initial keyword list) appearing in our data are trade, Trade, deal, China, and Xi (products and manufacturing not far behind). CCP, Communist Party, and Beijing do not show up that frequently. In the future, we will probably employ functionality to search for the most frequently appearing useful words (excludes articles, punctuation, etc.). The histogram is reproduced below:

<div class="output_png output_subarea ">
<img src="output_13_0.png">
</div>	

Next, we examine the VIX data. Below are summary statistics for the data we were able to pull and consolidate from 
	Bloomberg.


<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Last Price</th>
      <th>price_delta</th>
      <th>price_delta_5</th>
      <th>24_hr_perc</th>
      <th>7_day_perc</th>
      <th>14_day_perc</th>
      <th>30_day_perc</th>
      <th>52_week_high</th>
      <th>52_week_perc</th>
      <th>vix_delta_sign</th>
      <th>year</th>
      <th>month</th>
      <th>hour</th>
      <th>minutes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>722237.000000</td>
      <td>722237.000000</td>
      <td>722237.000000</td>
      <td>721833.000000</td>
      <td>719409.000000</td>
      <td>716581.000000</td>
      <td>710294.000000</td>
      <td>676514.000000</td>
      <td>676514.000000</td>
      <td>722236.000000</td>
      <td>722237.000000</td>
      <td>722237.000000</td>
      <td>722237.000000</td>
      <td>722237.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.738460</td>
      <td>-0.000022</td>
      <td>-0.000087</td>
      <td>0.001939</td>
      <td>0.010396</td>
      <td>0.016207</td>
      <td>0.022988</td>
      <td>38.205824</td>
      <td>0.399478</td>
      <td>0.000267</td>
      <td>2017.472323</td>
      <td>6.566354</td>
      <td>9.413017</td>
      <td>594.44071</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.103039</td>
      <td>0.062353</td>
      <td>0.125227</td>
      <td>0.067065</td>
      <td>0.163789</td>
      <td>0.218262</td>
      <td>0.275639</td>
      <td>11.517582</td>
      <td>0.103463</td>
      <td>0.355108</td>
      <td>1.113672</td>
      <td>3.335537</td>
      <td>3.796773</td>
      <td>226.93971</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.910000</td>
      <td>-12.690000</td>
      <td>-12.730000</td>
      <td>-0.430000</td>
      <td>-0.560000</td>
      <td>-0.610000</td>
      <td>-0.610000</td>
      <td>17.280000</td>
      <td>0.210000</td>
      <td>-1.000000</td>
      <td>2015.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>195.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.910000</td>
      <td>-0.010000</td>
      <td>-0.030000</td>
      <td>-0.030000</td>
      <td>-0.080000</td>
      <td>-0.100000</td>
      <td>-0.130000</td>
      <td>26.720000</td>
      <td>0.330000</td>
      <td>0.000000</td>
      <td>2017.000000</td>
      <td>4.000000</td>
      <td>6.000000</td>
      <td>395.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.680000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.010000</td>
      <td>-0.020000</td>
      <td>-0.020000</td>
      <td>36.200000</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>2017.000000</td>
      <td>7.000000</td>
      <td>10.000000</td>
      <td>610.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16.580000</td>
      <td>0.010000</td>
      <td>0.030000</td>
      <td>0.030000</td>
      <td>0.070000</td>
      <td>0.090000</td>
      <td>0.110000</td>
      <td>50.300000</td>
      <td>0.450000</td>
      <td>0.000000</td>
      <td>2018.000000</td>
      <td>9.000000</td>
      <td>13.000000</td>
      <td>791.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>50.200000</td>
      <td>8.570000</td>
      <td>8.520000</td>
      <td>1.490000</td>
      <td>2.570000</td>
      <td>3.320000</td>
      <td>3.920000</td>
      <td>53.290000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2019.000000</td>
      <td>12.000000</td>
      <td>16.000000</td>
      <td>974.00000</td>
    </tr>
  </tbody>
</table>
</div>
</div>

<p>Just based on the above, we can see a pretty large range in the values of the VIX, 
	suggesting that the pricing jumps around a lot. At the same time, though, we see that on a minute by minute basis,
	the price change is very small. We keep this in mind when developing our models because even though are outcome is price
	delta, we experiment with different time intervals over which that delta is calculated to see which model will be most 
	useful. </p>
	  
<p>Additionally, in our EDA, we looked at a snippet of the VIX pricing data. Our thought process was before diving into 
	any analysis, we should first determine when the VIX rose or fell significantly and see if those changes appear 
	to be related to any significant news events around those times. Our initial graph of the VIX data looks at EOD VIX 
	pricing over a set period of time (December 1, 2015 - December 1, 2016). (Our analysis will hone in on minute by 
	minute, but for us, it is important to be aware of the major surges). This graph is not reproduced below because of 
	its size, but it can be seen in the notebook (title: “VIX Pricing over time”). The following were the main surges 
	during this time period:</p>
	
<ul>
	<li>12/11/15</li>
	<li>1/19/16</li>
	<li>2/11/16</li>
	<li>6/14/16</li> 
	<li>6/27/16</li>
	<li>9/12/16</li>
	<li>Two weeks before 2016 Presidential elections and a week afterwards</li>
</ul>
	
<p>For each date, we cross referenced it with the Financial Times Archives for the few days before and a few days after. 
Note: VIX is incredibly volatile, where for most metrics/indices a 3-4% change on any given day is regarded as a big move, 
such changes are the norm for VIX. Thus explaining any major surge due to one event remains challenging, yet for a high 
level understanding, we believe it to be necessary.</p>

<ul>
	<li>Oil Prices Reaching Seven Year Lows</li>
	<li>Wall St. makes worst start to year, global bearishness, oil resumes slide, Fed might raise interest rates again</li>
	<li>Very mixed news, articles suggest renewed China risks due to China’s unruly peer-to-peer lending — 21 people arrested involved in “a complete Ponzi scheme” — ballooned in size last year as credit-starved private companies paid swingeing interest rates to secure loans </li>
	<li>Weak jobs data, major uncertainty regarding Fed hiking interest rates</li> 
	<li>Post-Brexit jitters</li>
	<li>Oil price hikes, Merkel loses major state election in response to her open-door refugee policy</li>
	<li>Three major moves: first, as polling data shows narrowing of Clinton’s lead, the VIX continuously climbs, one of its more significant spikes being the day Fed reopened the investigation into her emails, secondly, one of VIX’s more significant downturns (though, still mild relative to the climbs of the previous weeks) was when the investigation was officially closed, and thirdly when Trump won, the VIX surged upwards, only to calm the day after when trading resumed, which was the exact trend following the Brexit vote. </li>
</ul>

<p>Note that the moves above are not directly related to Trump's tweets, but they did give us a sense for how striking news could affect VIX pricing on a minute by minute basis.</p>

<p><img src="output_25_0.png"></p>





