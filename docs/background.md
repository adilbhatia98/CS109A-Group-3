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

# Motivation

Spotify is a popular music streaming platform that allows users to curate their own playlists. 
Music recommender systems help users find songs based on what they might like. 
We wanted to help Grace curate a Spotify playlist of her favorite songs using the classification techniques we learned in class.
Grace struggles to classify with words what makes a song fit into her vibe. 
"I'd jam to Madison Thompson's 'Lonely Together,' but definitely not to Avicii's."
When words fail, data science might help. 
We decided to use techniques from Cs109a to solve the problem of finding the best classification model for Grace's playlist, 
and then use that model to find her more songs for her playlist.

We had three goals:
1. to create the best performing model to classify songs as in or out of Grace's playlist
2. to use that model to suggest new songs for Grace's playlist
3. to produce a sleek interface using <i>jekyll</i> to display our analysis, so that others can replicate our method on their playlists

# Approach
First, we asked Grace to label a set of songs as included in her playlist and another set of songs as songs she would not want to listen to.
We used Spotify API to download her playlists with feature data for each song in the playlist.
We then randomly split this data into a training and test set. 
Next, we built and fit a variety of classifiers for Grace's playlist on the training set. Models we built include:
- Baseline Model
- Logistic Classifier
- Logistic Classifier With Quadratic Terms
- Logistic Classifier With L1 Regularization
- Logistic Classifier With L2 Regularization
- k-Nearest Neighbors
- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Decision Tree Classifier
- Decision Tree Classifier with Bagging
- Random Forest
- Decision Tree Classifier with Boosting
- Neural Network

For each model, we evaluated its accuracy on both our training and and test set. 
Based on accuracy scores, we determined the classifier with the highest performance on the test set. 
Finally, we ran our best-performing model on a fresh set of songs and asked Grace if she liked her new playlist.

---


# Literature Review 
[Current Challenges and Visions in Music Recommender Systems Research](https://arxiv.org/pdf/1710.03208.pdf)
+ Biggest current issues in MRS: cold start, automatic playlist generation, and evaluation.
+ State-of-the-art techniques in playlist extension include collaborative filtering and Markov chain models
	- limitations include ordering of songs within playlists and incorporating situational characteristics that affect listeners
+ Future work includes incorporating personality, current emotional state, political situation, and cultural situation into music recommendations.

[An Analysis of Approaches Taken in the ACM RecSys Challenge 2018](https://arxiv.org/pdf/1810.01520.pdf)
+ In 2018, Spotify sponsored a challenge involving addressing the automatic music playlist continuation problem
+ Most accurate classifiers involved:
	- Larger training sets produced better-performing models
		- If training sets were subset, using a random subset of the playlist rather than the sequentially first songs in a playlist was most accurate in training a model
	- Excluding "title" as meta-data for the playlist produced better models across the board
	- Solutions using the descriptors from the Spotify API were more efficient
	
	
	
	
	

[TrailMix: An Ensemble Recommender System for Playlist Curation and Continuation](people.tamu.edu/~zhaoxing623/publications/XZ_TrailMix.pdf)
+ Successful RecSys project TrailMix compared 3 recommender models:
	- song clustering purely based on title
	- decorated neural collaborative filtering
	- decision tree
+ The title model performed very poorly compared to the other two models
	- The authors acknowledge that using analysis beyond the literal words, ex. incorporating NLP methods, could help
+ An ensemble of all three models performed best

[Ways Explanations Impact End Users’ Mental Models](http://openaccess.city.ac.uk/6344/3/VLHCC2013.pdf)
+ This paper explored the explanations of music song recommenders to users
+ Part of enabling users to debug their models is explaining these agents to users well enough for them to build useful mental models
+ A kNN and bagged decision tree ensemble model was used to recommend songs
+ Completeness of explanations were more important than soundness in understanding models (ie--explaining everything as opposed to explaining only a part but explaining that part fully correctly)
+ They found that in general, comprehensive explanations were more useful than simple ones

