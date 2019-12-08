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
We pulled all of Trump's tweets in the last few years from his Twitter archive. Due to the naature of the database, we spent much time cleaning this data for our purposes. We took the following steps to clean the data:</p>
+ Importing raw data from the archive
+ Removing unnecessary columns
+ Adjusting GMT to Eastern Time + accounting for daylight savings
+ Manually fixing errors in cells where the delimiting was incorrectly done in the database output and manually re-inserting the delimiting character
	
We utilized `ntlk</code`'s `textblob` function in order to analyze the sentiment of tweets in our data set. For each tweet, this function created a polarity score (the more positive a tweet is, the closer the score is to 1; the more negative, the closer it is to -1). The function also returns a subjectivity score. Lower subjectivity score means that the tweet's polarity score more objectively represents its sentiment.

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


We have 5060 songs in our initial analysis. 2650 are included in Grace's playlist, and 2500 are not included in Grace's playlist. 

Below are summary statistics for all the features we plan to analyze:


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
      <th>feature</th>
      <th>mean</th>
      <th>var</th>
      <th>range</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>acousticness</td>
      <td>0.540199</td>
      <td>1.267884e-01</td>
      <td>9.959953e-01</td>
      <td>0.000005</td>
      <td>0.996</td>
    </tr>
    <tr>
      <td>danceability</td>
      <td>0.570920</td>
      <td>2.931912e-02</td>
      <td>9.162000e-01</td>
      <td>0.061800</td>
      <td>0.978</td>
    </tr>
    <tr>
      <td>duration_ms</td>
      <td>245718.492885</td>
      <td>1.911563e+10</td>
      <td>3.346533e+06</td>
      <td>44507.000000</td>
      <td>3391040.000</td>
    </tr>
    <tr>
      <td>energy</td>
      <td>0.439224</td>
      <td>6.633419e-02</td>
      <td>9.901450e-01</td>
      <td>0.000855</td>
      <td>0.991</td>
    </tr>
    <tr>
      <td>instrumentalness</td>
      <td>0.143138</td>
      <td>9.302492e-02</td>
      <td>9.870000e-01</td>
      <td>0.000000</td>
      <td>0.987</td>
    </tr>
    <tr>
      <td>key</td>
      <td>5.223913</td>
      <td>1.251578e+01</td>
      <td>1.100000e+01</td>
      <td>0.000000</td>
      <td>11.000</td>
    </tr>
    <tr>
      <td>liveness</td>
      <td>0.163377</td>
      <td>1.798945e-02</td>
      <td>9.800000e-01</td>
      <td>0.012000</td>
      <td>0.992</td>
    </tr>
    <tr>
      <td>loudness</td>
      <td>-10.270219</td>
      <td>3.464989e+01</td>
      <td>4.217600e+01</td>
      <td>-42.476000</td>
      <td>-0.300</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>0.650198</td>
      <td>2.274856e-01</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>popularity</td>
      <td>36.977470</td>
      <td>4.773025e+02</td>
      <td>1.000000e+02</td>
      <td>0.000000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <td>speechiness</td>
      <td>0.070655</td>
      <td>6.217856e-03</td>
      <td>8.989000e-01</td>
      <td>0.023100</td>
      <td>0.922</td>
    </tr>
    <tr>
      <td>tempo</td>
      <td>117.657563</td>
      <td>8.604272e+02</td>
      <td>1.790410e+02</td>
      <td>42.581000</td>
      <td>221.622</td>
    </tr>
    <tr>
      <td>time_signature</td>
      <td>3.919763</td>
      <td>1.655315e-01</td>
      <td>5.000000e+00</td>
      <td>0.000000</td>
      <td>5.000</td>
    </tr>
    <tr>
      <td>valence</td>
      <td>0.425801</td>
      <td>5.455384e-02</td>
      <td>9.591000e-01</td>
      <td>0.025900</td>
      <td>0.985</td>
    </tr>
  </tbody>
</table>
</div>


We can see that all features have values that are expected as per the Spotify API documentation. 
To analyze each feature in more granularity we looked at density plots. 


![png](output_18_0.png)


Looking at the density plots above, we note some features that show clear differences in distribution between the playlist and non-playlist. 
While non-playlist songs contain a roughly uniform distribution of energy values, playlist songs spike at an energy level between 0.2-0.4.
Acousticness in playlist tracks is much higher on average, spiking around 0.8, while non-playlist tracks most frequently have acousticness values around 0.1.
Instrumentalness is a particularly interesting feature. While the distribution non-playlist tracks is bimodal, peaking at around 0 and 0.9, playlist tracks have a few very well-defined peaks between 0 and 0.3. 
We will note in advance that this may induce a risk of overfitting based on instrumentalness values.
Playlist tracks have lower loudnesses on average, centering around -10, while non-playlist tracks -5.
In terms of speechiness, the distribution for playlist tracks has a much lower variance and slightly lower expected value, centering around 0.3 while non-playlist tracks center around 0.4.
Valence for non-playlist tracks is roughly uniformly distributed, while playlist tracks demonstrate a roughly normal distribution centered around 0.3.
Finally in terms of popularity, playlist tracks show a peak in their distribution around 60, while non-playlist tracks have a more variable distribution with a peak between 45-55.
The rest of the features are roughly similar in distribution between playlist and non-playlist tracks.


![png](output_20_1.png)


The pairplot above demonstrates a few interesting things. First, we notice weakly positive correlations between loudness and energy, loudness and danceability, and danceablility and loudness. We also notice a negative correlation between acousticness and energy. These correlations will be useful to keep in mind if we conduct variable selection or regularization at a later point. Also, none of these pairwise plots show clear separability between the two playlists.





