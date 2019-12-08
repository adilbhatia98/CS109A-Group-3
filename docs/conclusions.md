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

# Analysis of Results
Our best model is the boosted decision tree classifier with a depth of 2 and 751 iterations, 
which performs with an accuracy of 95.4% in the training set and 93.0% in the test set.

The following table summarizes the accuracies for all our models, ordered by accuracy in the test set:

|                 Model Type                 | Train Accuracy      | Test Accuracy      |
|--------------------------------------------|:-------------------:|:------------------:|
|   Naive Model  							 |       50.3%         |       51.6%        |
|                 Neural Network             |       62.8%         |       65.2%        |
|                     kNN                    |       63.1%         |       65.9%        |
| Logistic Regression With L2 Regularization |       69.2%         |       66.9%        |
|        Baseline Logistic Regression        |       69.4%         |       67.1%        |
|                     QDA                    |       86.6%         |       86.7%        |
|                     LDA                    |       88.1%         |       88.4%        |
| Logistic Regression With L1 Regularization |       88.6%         |       88.7%        |
|          Decision Tree Classifier          |       88.0%         |       89.0%        |
|    Decision Tree Classifier With Bagging   |       93.6%         |       91.8%        |
|               Random Forest                |       92.9%         |       92.0%        |
|       Boosted Decision Tree Classifier     |       95.4%         |       93.0%        |

Our lowest performing models include the logistic regression with the naive model, the neural network, and the kNN model.
Our best performing models were all ensemble methods. 
The boosted decision tree classifier, the random forest model, and the decision tree classifier with bagging performed best.
We tuned the parameters and hyperparameters of each base model to maximize the accuracy score of each, 
which leads us to believe that we achieved the maximum possible classification accuracy given the constraints of our dataset.
Indeed, this model performs much better than the baseline model, which achieves an accuracy of only 51.6% in the test set.

Finally, while usually time and space are considerations when evaluating different types of models, 
because they do not constrain our original problem, we chose to focus on accuracy. 
However, a qualitative assessment of these metrics determined that all models were comparable in terms of runtime and memory use with the exception of the neural nets that took additional time.
One consideration to note is that the boosted model is less suited for parallelization than other ensemble methods, because it is iterative. However, the runtime was less than a few seconds, so we will prioritize the accuracy of the model over this concern.
---

# Extending Our Model
We can now try to generate a playlist customized to Grace's taste using our chosen model. 
We will present the model with a list of songs that both Grace and the model have not seen before. 
We'll then have the model assess whether these songs should be included in the playlist and then verify that with Grace's opinion.


```python
# load in dataset
full_songs_df = pd.read_csv("data/spotify-test.csv")

# drop unnecessary columns
songs_df = full_songs_df.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url', 'name', 'artist', 'Unnamed: 0'])
```


```python
# recreating the best model
best_abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=800, learning_rate = 0.05)
best_abc.fit(x_train, y_train)
predictions = best_abc.predict(songs_df)
```


```python
print("Songs Selected for Grace's Playlist")
for i in range(len(predictions)):
    if predictions[i] == 1:
        print(full_songs_df.loc[i]['name'])
```

    Songs Selected for Grace's Playlist
    Never Seen Anything "Quite Like You" - Acoustic
    Crazy World - Live from Dublin
    Whatever You Do
    Come on Love
    1,000 Years
    Machine
    After Dark
    Sudden Love (Acoustic)
    Georgia
    I Don't Know About You


This randomly selected dataset had 26 songs. 
These songs had never been classified by Grace before and our best model (the boosted decision tree classifier with a depth of 2) was used to predict whether songs would be included in her playlist. We then played all the songs in the dataset to Grace to see whether she would include them in her playlist. 
The model performed accurately, except for one song which she said she would not have added to her playlist ("I Don't Know About You"). 
One reason for this mishap could be that our model isn't 100% accurate, so this song could be by chance one of the ones it messes up; 1 missed song out of 26 is reasonable for a model with 93% accuracy. 
Another reason could be that Grace's actual taste is different from how she made the playlist (perhaps she is in a different emotive or environmental state that temporally affects her preferences, or perhaps her underlying preferences have changed). 
Despite this error, overall, Grace was pleased that we could use data science to automate her playlist selection procees!

---


# Limitations and Future Work
### Data Size
{: .no_toc }
We generated a dataset by consolidating a large array of songs that vary in genre, language, tempo, rhythm, etc. We tried to curate a dataset that mimicked the variety of songs that Spotify has. Grace then had to go through these songs and classify whether she would like them in her playlist or not. Due to a multitude of constraints, we only had 5000 songs between both and training and test data. Ideally more songs that accurately capture the variety of songs that Spotify has would improve the training procedures for models.

### Data Inclusion
{: .no_toc }
Outside of the side of the data set, there are other data sets that can be used discover new predictors or variables about songs. We can explore lyrics for example and see how that contributes to a model's recommendations. Thus requires us expanding beyond the SpotifyAPI and exploring other data sets.

### Adapting Playlists 
{: .no_toc }
This entire project was built off of the preferences of one individual: Grace. While this proved to be a good proof of concept, future exploration should be done to analyze how the best model can help create playlists for others based upon their interests. 

### Collaborative Filtering
{: .no_toc }
Collaborative filtering is another type of data modeling that is commonly used for recommendation algorithms. It is based on the fundamental idea that people perfer things similar to the things they've established they like. As such, it would be a good model to further investigate for this given project. 

### Improve Neural Network
{: .no_toc }
Our neural network did not perform particularly well.
While we tuned many hyperparameters, further tuning, exploring other network structures, and changing optimizers may help improve our network.
Additionally, we could consider using convolutional neural networks.

### Dynamic Preferences
{: .no_toc }
Finally, a review of the literature highlighted the importance of dynamic preferences.
Individuals often adapt the music they want to listen to at a particular time based on their emotions or external situation.
Allowing for a modification of models to include these parameters could be useful.
For example, if Grace tracked the time of day that she added each song to her playlist and we could use that time as a feature, we could better suggest songs suited to a particular time of day.
This could help adjust for temporal effects, such as desiring certain music during a morning run or commute to work, versus desiring different music for an evening shower, versus desiring different music for a party late at night.


