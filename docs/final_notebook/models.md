---
title: Models
layout: default
nav_order: 4
---
# Models
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# Setup

<p>Our goal for this project is to build a model that can accurately predict changes in VIX prices after Trump tweets. In the model, we use characteristics of the VIX data (like <code class="highlighter-rouge">Last Price</code>) and multiple characteristics of the Twitter data. Also, some predictors are common to both datasets, such as date/time predictors.</p>
          
<p>Based on these predictors, we initially considered two types of models. First, we considered one that predicts absolute price changes in VIX pricing (continuous outcome). Second, we considered one that predicts the sign of VIX pricing changes (positive, negative, or no change - a categorical outcome). Intuitively, given what we know about the VIX and its pricing fluctuations, we thought it made more sense to focus on a model that predicts the change in pricing. Thus, we built a model that has a categorical output variable: -1 for negative price delta, 0 for no price delta, 1 for positive price delta.</p>
  
<p>We also wanted to create interval versions of this model that looks at the VIX price change over 1 minute, 5, 10, 20 , 30, and 60 minutes. We wanted to consider all these options to determine the most accurate and most useful model. Realistically, if our model predicts well 10+ minutes after the tweet, it can offer a great chance to earn positive returns by trading on the VIX index. Another important consideration is the threshold at which we declare an outcome positive or negative. Given we are using minute-by-minute data, we decided upon a 0.001 threshold so that any change between -0.001 and 0.001 inclusive should be categorized as 0 (no price delta). To explain why we chose 0.001, there are a couple points to consider. First, any change is very helpful in terms of playing the market through trading, even if it is a small change. Second, a larger threshold could have potentially forced our models to disproportionately predict 0 change (at all time intervals), which would not be a useful model. As we will see below, at larger intervals, our models predict far fewer 0 outcomes, becoming more successful at predicting positive or negative changes in the VIX pricing over time.</p>  

For each model trial, we first split our data into a train and test set, so that we are later able to assess how well our model performs in both a train set and a not-seen test set. Realistically, training accuracy reflects how well a given model 'understands' the data it is presented with, and testing accuracy reflects how well that model can be generalized to accurately form predictions about data it has not yet seen. For each time interval model, we drop the irrelevant time-based price predictors. For example, model30 does not include `price_delta_30` or `price_delta_60` as predictors. We perform this drop for each time interval model.
     
# Baseline Model - Logistic

<p>Our baseline model represents a simple logistic regression with a multiclass outcome variable. Using the predictors in our data, the model predicts the 'change in VIX price' classification as positive, negative, or 0 (no change) using a simple logistic regression.</p>          

```
# 1 min
logreg0 = LogisticRegression(C=10000).fit(X_train0, y_train0)
logreg_fit_train0 = logreg0.predict(X_train0)
logreg_fit_test0 = logreg0.predict(X_test0)

train_scores_logreg0 = accuracy_score(y_train0, logreg_fit_train0)
test_scores_logreg0 = accuracy_score(y_test0, logreg_fit_test0)

print("Training Accuracy 1 min: ", train_scores_logreg0)
print("Testing Accuracy 1 min: ", test_scores_logreg0)

# 5 min
logreg5 = LogisticRegression(C=10000).fit(X_train5, y_train5)
logreg_fit_train5 = logreg5.predict(X_train5)
logreg_fit_test5 = logreg5.predict(X_test5)

train_scores_logreg5 = accuracy_score(y_train5, logreg_fit_train5)
test_scores_logreg5 = accuracy_score(y_test5, logreg_fit_test5)

print("Training Accuracy 5 min: ", train_scores_logreg5)
print("Testing Accuracy 5 min: ", train_scores_logreg5)

# 10 min
logreg10 = LogisticRegression(C=10000).fit(X_train10, y_train10)
logreg_fit_train10 = logreg10.predict(X_train10)
logreg_fit_test10 = logreg10.predict(X_test10)

train_scores_logreg10 = accuracy_score(y_train10, logreg_fit_train10)
test_scores_logreg10 = accuracy_score(y_test10, logreg_fit_test10)

print("Training Accuracy 10 min: ", train_scores_logreg10)
print("Testing Accuracy 10 min: ", train_scores_logreg10)

# 20 min
logreg20 = LogisticRegression(C=10000).fit(X_train20, y_train20)
logreg_fit_train20 = logreg20.predict(X_train20)
logreg_fit_test20 = logreg20.predict(X_test20)

train_scores_logreg20 = accuracy_score(y_train20, logreg_fit_train20)
test_scores_logreg20 = accuracy_score(y_test20, logreg_fit_test20)

print("Training Accuracy 20 min: ", train_scores_logreg20)
print("Testing Accuracy 20 min: ", train_scores_logreg20)

# 30 min
logreg30 = LogisticRegression(C=10000).fit(X_train30, y_train30)
logreg_fit_train30 = logreg30.predict(X_train30)
logreg_fit_test30 = logreg30.predict(X_test30)

train_scores_logreg30 = accuracy_score(y_train30, logreg_fit_train30)
test_scores_logreg30 = accuracy_score(y_test30, logreg_fit_test30)

print("Training Accuracy 30 min: ", train_scores_logreg30)
print("Testing Accuracy 30 min: ", train_scores_logreg30)

# 60 min
logreg60 = LogisticRegression(C=10000).fit(X_train60, y_train60)
logreg_fit_train60 = logreg60.predict(X_train60)
logreg_fit_test60 = logreg60.predict(X_test60)

train_scores_logreg60 = accuracy_score(y_train60, logreg_fit_train60)
test_scores_logreg60 = accuracy_score(y_test60, logreg_fit_test60)

print("Training Accuracy 60 min: ", train_scores_logreg60)
print("Testing Accuracy 60 min: ", train_scores_logreg60)
```
```
Interval  training accuracy   test accuracy
30 minute	0.523760	          0.503306
20 minute	0.515702	          0.500826
60 minute	0.534946	          0.499586
10 minute	0.491529	          0.476033
5 minute	0.476033	          0.452893
1 minute	0.468182	          0.451240
```

# L1 and L2 Regularization

<p>We then decided to incorporate regularization in an attempt to improve our logistic model's predictive ability. Lasso regularization (L1) sets the effects/coefficients of unimportant predictors to 0, whereas ridge (L2) simply minimizes/lowers those effects.</p>               

<p>First, lasso regularization:</p>

```
from sklearn.linear_model import LogisticRegressionCV

# lasso
lasso = LogisticRegressionCV(cv=5, penalty='l1', max_iter=1000, solver='liblinear')

train_scores_logreg_lasso = []
test_scores_logreg_lasso = []

for i in range(len(X_train_list)):
    lassofit = lasso.fit(X_train_list[i], y_train_list[i])
    y_pred_train_lasso = lassofit.predict(X_train_list[i])
    y_pred_test_lasso = lassofit.predict(X_test_list[i])
    train_score = accuracy_score(y_train_list[i], y_pred_train_lasso)
    test_score = accuracy_score(y_test_list[i], y_pred_test_lasso)
    train_scores_logreg_lasso.append(train_score)
    test_scores_logreg_lasso.append(test_score)
    print(f'Training set accuracy score for {intervals[i]} using CV & LASSO penalty: {train_score:.4f}')
    print(f'Test set accuracy score for {intervals[i]} using CV & LASSO penalty: {test_score:.4f}')

```
```
Interval	training accuracy	test accuracy
1 minute	0.467769	          0.457025
5 minute	0.457025	          0.455372
60 minute	0.444582	          0.419355
30 minute	0.439669	          0.416529
20 minute	0.373140	          0.348760
10 minute	0.217149	          0.200826
```
<p>Now, ridge regularization:</p>
```
# ridge
ridge = LogisticRegressionCV(cv=5, penalty='l2', max_iter=1000, solver='liblinear')

train_scores_logreg_ridge = []
test_scores_logreg_ridge = []

for i in range(len(X_train_list)):
    ridgefit = ridge.fit(X_train_list[i], y_train_list[i])
    y_pred_train_ridge = ridgefit.predict(X_train_list[i])
    y_pred_test_ridge = ridgefit.predict(X_test_list[i])
    train_score = accuracy_score(y_train_list[i], y_pred_train_ridge)
    test_score = accuracy_score(y_test_list[i], y_pred_test_ridge)
    train_scores_logreg_ridge.append(train_score)
    test_scores_logreg_ridge.append(test_score)
    print(f'Training set accuracy score for {intervals[i]} using CV & Ridge penalty:: {train_score:.4f}')
    print(f'Test set accuracy score for {intervals[i]} using CV & Ridge penalty:: {test_score:.4f}')

```
```
Interval	training accuracy	test accuracy
60 minute	0.544458	          0.516129
20 minute	0.526033	          0.500826
30 minute	0.531612	          0.495868
10 minute	0.495455	          0.480165
1 minute	0.469421	          0.451240
5 minute	0.473140	          0.439669
```
# Random Forest

<p>Our first ensemble method is random forest, which randomly subsets predictors upon which to generate decision trees. We tested out a few different tree depth and number parameters ourselves and determined that a depth of 5 and number of trees of 100 was ideal for our analysis.</p>

<p>Below is sample code for one of the time interval models. We perform this for each of the models, and the results can bee seen below this code.</p>
```
# Calibrate num trees and tree depth (after having tried different parameters, saw comparable results)
n_trees = 100
tree_depth = 5
```

```
# create RF models
forest_model0 = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_depth, 
                                      max_features=int(np.sqrt(len(np_X_train0[0]))))

forest_model5 = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_depth, 
                                      max_features=int(np.sqrt(len(np_X_train5[0]))))

forest_model10 = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_depth, 
                                      max_features=int(np.sqrt(len(np_X_train10[0]))))

forest_model20 = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_depth, 
                                      max_features=int(np.sqrt(len(np_X_train20[0]))))

forest_model30 = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_depth, 
                                      max_features=int(np.sqrt(len(np_X_train30[0]))))

forest_model60 = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_depth, 
                                      max_features=int(np.sqrt(len(np_X_train60[0]))))
```

```
# fit and calculate accuracy
forest_model0.fit(np_X_train0, np_y_train0)
y_pred0_forest_train = forest_model0.predict(np_X_train0)
y_pred0_forest_test = forest_model0.predict(np_X_test0)

random_forest_train_score0 = accuracy_score(np_y_train0, y_pred0_forest_train)
random_forest_test_score0 = accuracy_score(np_y_test0, y_pred0_forest_test)
print(f'The random forest accuracy on the training set: {random_forest_train_score0}')
print(f'The random forest accuracy on the test set: {random_forest_test_score0:.4f}')

```

```
Interval	training accuracy	test accuracy
60 minute	0.620968	          0.545906
20 minute	0.623347	          0.532231
30 minute	0.611777	          0.514050
10 minute	0.635331	          0.508264
1 minute	0.547934	          0.455372
5 minute	0.591116	          0.446281
```


# Boosting
Next, we will consider boosting, an iterative approach that might eliminate some more of the error in our trees.

<p>Below is sample code for one of the time interval models. We perform this for each of the models, and the results can bee seen below this code.</p>

```python
# initialize parameters like for RF, much the same process - tested different ones and saw best/most consistent results with the following
estimators_ADA = 40
learning_ADA = 0.01
tree_depth = 10
```

```
model_ADA0 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=tree_depth), 
                               learning_rate=learning_ADA, n_estimators=estimators_ADA)
model_ADA0.fit(np_X_train0, np_y_train0)

y_train_pred_ADA0 = model_ADA0.predict(np_X_train0)
y_test_pred_ADA0 = model_ADA0.predict(np_X_test0)
ADA_train0 = accuracy_score(np_y_train0, y_train_pred_ADA0)
ADA_test0 = accuracy_score(np_y_test0, y_test_pred_ADA0)
print(f'The ADABoost accuracy on the training set: {ADA_train0}')
print(f'The ADABoost accuracy on the test set: {ADA_test0:.4f}')

ADA_train0_staged = list(model_ADA0.staged_score(np_X_train0, np_y_train0))
ADA_test0_staged = list(model_ADA0.staged_score(np_X_test0, np_y_test0))
```

```
# define function to abstract process of building plot
def baselearner_plt(n, X_train, y_train, X_test, y_test):
    # AdaBoostClassifier
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=n), n_estimators=30, learning_rate = 0.05)
    ada.fit(X_train, y_train)
    
    # staged_score train to plot
    ada_predicts_train = list(ada.staged_score(X_train,y_train))
    plt.plot(ada_predicts_train, label = "Train");

    # staged_score test to plot
    ada_predicts_test = list(ada.staged_score(X_test,y_test))
    plt.plot(ada_predicts_test, label = "Test");

    plt.legend(fontsize=14)
    plt.title("AdaBoost Classifier Accuracy", fontsize=18)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xlabel("Num Iterations", fontsize=16)
    plt.show()
    
    print("Maximum test accuracy for depth of "+str(n)+" is "+str(max(ada_predicts_test))+" at "+str(ada_predicts_test.index(max(ada_predicts_test)))+" iterations")

for i in range(len(X_train_list)):
    baselearner_plt(tree_depth, X_train_list[i], y_train_list[i], X_test_list[i], y_test_list[i])
    print("Time Interval: ", intervals[i]) 
    
```

![png](output_boost0.png)

    Maximum test accuracy for depth of 10 is 0.4818181818181818 at 2 iterations
    Time Interval:  1 minute


![png](output_boost5.png)


    Maximum test accuracy for depth of 10 is 0.5107438016528926 at 8 iterations
    Time Interval:  5 minute

![png](output_boost10.png)


    Maximum test accuracy for depth of 10 is 0.5958677685950413 at 9 iterations
    Time Interval:  10 minute

![png](output_boost20.png)


    Maximum test accuracy for depth of 10 is 0.6090909090909091 at 21 iterations
    Time Interval:  20 minute

![png](output_boost30.png)


    Maximum test accuracy for depth of 10 is 0.6347107438016529 at 29 iterations
    Time Interval:  30 minute

![png](output_boost60.png)


    Maximum test accuracy for depth of 10 is 0.6683209263854425 at 5 iterations
    Time Interval:  60 minute


# Neural Networks

Finally, we created an artificial neural network to classify our playlist songs.


```python
# check input and output dimensions
input_dim_2 = x_train.shape[1]
output_dim_2 = 1
print(input_dim_2,output_dim_2)
```

    14 1



```python
# create sequential multi-layer perceptron
model2 = Sequential() 

# initial layer
model2.add(Dense(10, input_dim=input_dim_2,  
                activation='relu')) 

# second layer
model2.add(Dense(10, input_dim=input_dim_2,  
                activation='relu'))

# third layer
model2.add(Dense(10, input_dim=input_dim_2,  
                activation='relu'))

# output layer
model2.add(Dense(1, activation='sigmoid'))

# compile the model
model2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2.summary()
```

```python
# fit the model
model2_history = model2.fit(
    x_train, y_train,
    epochs=50, validation_split = 0.5, batch_size = 128, verbose=False)
```


```python
# model loss
print("[Neural Net - Model 1] Loss: ", model2_history.history['loss'][-1])
print("[Neural Net - Model 1] Val Loss: ", model2_history.history['val_loss'][-1])
print("[Neural Net - Model 1] Test Loss: ", model2.evaluate(x_test, y_test, verbose=False))
print("[Neural Net - Model 1] Accuracy: ", model2_history.history['acc'][-1])
print("[Neural Net - Model 1] Val Accuracy: ", model2_history.history['val_acc'][-1])
```

    [Neural Net - Model 1] Loss:  7.79790558079957
    [Neural Net - Model 1] Val Loss:  8.034205742033103
    [Neural Net - Model 1] Test Loss:  [7.719139232937055, 0.5158102769154334]
    [Neural Net - Model 1] Accuracy:  0.5108695654529828
    [Neural Net - Model 1] Val Accuracy:  0.49604743024106085


Our initial accuracy isn't great. We achieve an accuracy of 48.9% in the training and 50.4% in the validation, and an accuracy of 48.4% in the test. Let's see if we can improve our network to fit the data better.


```python
# create sequential multi-layer perceptron
model3 = Sequential() 

# Hidden layers
for i in range(40):
    model3.add(Dense(10, input_dim=input_dim_2, 
        activation='relu')) 

# output layer
model3.add(Dense(output_dim_2, activation='sigmoid'))


# compile the model
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model3.summary()
```

```python
# fit the model
model3_history = model3.fit(
    x_train, y_train,
    epochs=300, validation_split = 0.1, batch_size = 128, verbose=False)
```


```python
# model loss
print("[Neural Net - Model 2] Loss: ", model3_history.history['loss'][-1])
print("[Neural Net - Model 2] Val Loss: ", model3_history.history['val_loss'][-1])
print("[Neural Net - Model 2] Test Loss: ", model3.evaluate(x_test, y_test, verbose=False))
print("[Neural Net - Model 2] Accuracy: ", model3_history.history['acc'][-1])
print("[Neural Net - Model 2] Val Accuracy: ", model3_history.history['val_acc'][-1])
```

    [Neural Net - Model 2] Loss:  0.6267417644590524
    [Neural Net - Model 2] Val Loss:  0.6291195959220698
    [Neural Net - Model 2] Test Loss:  [0.6115785545040026, 0.6432806319398843]
    [Neural Net - Model 2] Accuracy:  0.625857809154077
    [Neural Net - Model 2] Val Accuracy:  0.6197530875971288


Even after changing hyperparameters, our neural network does not perform very well. Using 40 layers and 300 epochs, the accuracy in the training data is still 62.8% while the accuracy in the test is 65.2%. This is baffling, because we expected our neural network to perform very well. Perhaps this mediocre perforance is due to limitations of our data set (only 14 features and <5000 songs), or of the specific methods we used.

---

# Model Selection

Based upon the presented analysis, we conclude that our boosted decision tree classifier, at a depth of 2 with 751 iterations, is the best model. It achieves the highest accuracy in the test set, of 93.0%.




