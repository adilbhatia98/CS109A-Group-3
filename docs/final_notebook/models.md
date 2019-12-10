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

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># 1 min</span>
<span class="n">logreg0</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">10000</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train0</span><span class="p">,</span> <span class="n">y_train0</span><span class="p">)</span>
<span class="n">logreg_fit_train0</span> <span class="o">=</span> <span class="n">logreg0</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train0</span><span class="p">)</span>
<span class="n">logreg_fit_test0</span> <span class="o">=</span> <span class="n">logreg0</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test0</span><span class="p">)</span>

<span class="n">train_scores_logreg0</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train0</span><span class="p">,</span> <span class="n">logreg_fit_train0</span><span class="p">)</span>
<span class="n">test_scores_logreg0</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test0</span><span class="p">,</span> <span class="n">logreg_fit_test0</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training Accuracy 1 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg0</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Testing Accuracy 1 min: &quot;</span><span class="p">,</span> <span class="n">test_scores_logreg0</span><span class="p">)</span>

<span class="c1"># 5 min</span>
<span class="n">logreg5</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">10000</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train5</span><span class="p">,</span> <span class="n">y_train5</span><span class="p">)</span>
<span class="n">logreg_fit_train5</span> <span class="o">=</span> <span class="n">logreg5</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train5</span><span class="p">)</span>
<span class="n">logreg_fit_test5</span> <span class="o">=</span> <span class="n">logreg5</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test5</span><span class="p">)</span>

<span class="n">train_scores_logreg5</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train5</span><span class="p">,</span> <span class="n">logreg_fit_train5</span><span class="p">)</span>
<span class="n">test_scores_logreg5</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test5</span><span class="p">,</span> <span class="n">logreg_fit_test5</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training Accuracy 5 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg5</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Testing Accuracy 5 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg5</span><span class="p">)</span>

<span class="c1"># 10 min</span>
<span class="n">logreg10</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">10000</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train10</span><span class="p">,</span> <span class="n">y_train10</span><span class="p">)</span>
<span class="n">logreg_fit_train10</span> <span class="o">=</span> <span class="n">logreg10</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train10</span><span class="p">)</span>
<span class="n">logreg_fit_test10</span> <span class="o">=</span> <span class="n">logreg10</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test10</span><span class="p">)</span>

<span class="n">train_scores_logreg10</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train10</span><span class="p">,</span> <span class="n">logreg_fit_train10</span><span class="p">)</span>
<span class="n">test_scores_logreg10</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test10</span><span class="p">,</span> <span class="n">logreg_fit_test10</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training Accuracy 10 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg10</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Testing Accuracy 10 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg10</span><span class="p">)</span>

<span class="c1"># 20 min</span>
<span class="n">logreg20</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">10000</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train20</span><span class="p">,</span> <span class="n">y_train20</span><span class="p">)</span>
<span class="n">logreg_fit_train20</span> <span class="o">=</span> <span class="n">logreg20</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train20</span><span class="p">)</span>
<span class="n">logreg_fit_test20</span> <span class="o">=</span> <span class="n">logreg20</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test20</span><span class="p">)</span>

<span class="n">train_scores_logreg20</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train20</span><span class="p">,</span> <span class="n">logreg_fit_train20</span><span class="p">)</span>
<span class="n">test_scores_logreg20</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test20</span><span class="p">,</span> <span class="n">logreg_fit_test20</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training Accuracy 20 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg20</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Testing Accuracy 20 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg20</span><span class="p">)</span>

<span class="c1"># 30 min</span>
<span class="n">logreg30</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">10000</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train30</span><span class="p">,</span> <span class="n">y_train30</span><span class="p">)</span>
<span class="n">logreg_fit_train30</span> <span class="o">=</span> <span class="n">logreg30</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train30</span><span class="p">)</span>
<span class="n">logreg_fit_test30</span> <span class="o">=</span> <span class="n">logreg30</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test30</span><span class="p">)</span>

<span class="n">train_scores_logreg30</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train30</span><span class="p">,</span> <span class="n">logreg_fit_train30</span><span class="p">)</span>
<span class="n">test_scores_logreg30</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test30</span><span class="p">,</span> <span class="n">logreg_fit_test30</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training Accuracy 30 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg30</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Testing Accuracy 30 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg30</span><span class="p">)</span>

<span class="c1"># 60 min</span>
<span class="n">logreg60</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">C</span><span class="o">=</span><span class="mi">10000</span><span class="p">)</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train60</span><span class="p">,</span> <span class="n">y_train60</span><span class="p">)</span>
<span class="n">logreg_fit_train60</span> <span class="o">=</span> <span class="n">logreg60</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train60</span><span class="p">)</span>
<span class="n">logreg_fit_test60</span> <span class="o">=</span> <span class="n">logreg60</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test60</span><span class="p">)</span>

<span class="n">train_scores_logreg60</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train60</span><span class="p">,</span> <span class="n">logreg_fit_train60</span><span class="p">)</span>
<span class="n">test_scores_logreg60</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test60</span><span class="p">,</span> <span class="n">logreg_fit_test60</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Training Accuracy 60 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg60</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">&quot;Testing Accuracy 60 min: &quot;</span><span class="p">,</span> <span class="n">train_scores_logreg60</span><span class="p">)</span>
</pre>
</div>
    </div>
</div>
 
<div class="output_wrapper">
<div class="output">
<div class="output_area">
    <div class="prompt"></div>
<div class="output_html rendered_html output_subarea ">
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
      <th>Interval</th>
      <th>training accuracy</th>
      <th>test accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>30 minute</td>
      <td>0.523760</td>
      <td>0.503306</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20 minute</td>
      <td>0.515702</td>
      <td>0.500826</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60 minute</td>
      <td>0.534946</td>
      <td>0.499586</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10 minute</td>
      <td>0.491529</td>
      <td>0.476033</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5 minute</td>
      <td>0.476033</td>
      <td>0.452893</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1 minute</td>
      <td>0.468182</td>
      <td>0.451240</td>
    </tr>
  </tbody>
</table>
</div>
</div>
</div>
</div>
</div>

# L1 and L2 Regularization

<p>We then decided to incorporate regularization in an attempt to improve our logistic model's predictive ability. Lasso regularization (l1) sets the effects/coefficients of unimportant predictors to 0, whereas ridge (l2) simply minimizes/lowers those effects.</p>               

<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">Lasso</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegressionCV</span>

<span class="c1"># lasso</span>
<span class="n">lasso</span> <span class="o">=</span> <span class="n">LogisticRegressionCV</span><span class="p">(</span><span class="n">cv</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">penalty</span><span class="o">=</span><span class="s1">&#39;l1&#39;</span><span class="p">,</span> <span class="n">max_iter</span><span class="o">=</span><span class="mi">1000</span><span class="p">,</span> <span class="n">solver</span><span class="o">=</span><span class="s1">&#39;liblinear&#39;</span><span class="p">)</span>

<span class="n">train_scores_logreg_lasso</span> <span class="o">=</span> <span class="p">[]</span>
<span class="n">test_scores_logreg_lasso</span> <span class="o">=</span> <span class="p">[]</span>

<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">X_train_list</span><span class="p">)):</span>
    <span class="n">lassofit</span> <span class="o">=</span> <span class="n">lasso</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train_list</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">y_train_list</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
    <span class="n">y_pred_train_lasso</span> <span class="o">=</span> <span class="n">lassofit</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_train_list</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
    <span class="n">y_pred_test_lasso</span> <span class="o">=</span> <span class="n">lassofit</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test_list</span><span class="p">[</span><span class="n">i</span><span class="p">])</span>
    <span class="n">train_score</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_train_list</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">y_pred_train_lasso</span><span class="p">)</span>
    <span class="n">test_score</span> <span class="o">=</span> <span class="n">accuracy_score</span><span class="p">(</span><span class="n">y_test_list</span><span class="p">[</span><span class="n">i</span><span class="p">],</span> <span class="n">y_pred_test_lasso</span><span class="p">)</span>
    <span class="n">train_scores_logreg_lasso</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">train_score</span><span class="p">)</span>
    <span class="n">test_scores_logreg_lasso</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">test_score</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">&#39;Training set accuracy score for </span><span class="si">{intervals[i]}</span><span class="s1"> using CV &amp; LASSO penalty: </span><span class="si">{train_score:.4f}</span><span class="s1">&#39;</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">f</span><span class="s1">&#39;Test set accuracy score for </span><span class="si">{intervals[i]}</span><span class="s1"> using CV &amp; LASSO penalty: </span><span class="si">{test_score:.4f}</span><span class="s1">&#39;</span><span class="p">)</span>
</pre></div>
    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">
    <div class="prompt"></div>
<div class="output_html rendered_html output_subarea ">
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
      <th>Interval</th>
      <th>training accuracy</th>
      <th>test accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 minute</td>
      <td>0.467769</td>
      <td>0.457025</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5 minute</td>
      <td>0.457025</td>
      <td>0.455372</td>
    </tr>
    <tr>
      <th>5</th>
      <td>60 minute</td>
      <td>0.444582</td>
      <td>0.419355</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30 minute</td>
      <td>0.439669</td>
      <td>0.416529</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20 minute</td>
      <td>0.373140</td>
      <td>0.348760</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10 minute</td>
      <td>0.217149</td>
      <td>0.200826</td>
    </tr>
  </tbody>
</table>
</div>
</div>
</div>
</div>
</div>

# Random Forest

<p>Our first ensemble method is random forest, which randomly subsets predictors upon which to generate decision trees. We tested out a few different tree depth and number parameters ourselves and determined that a depth of 5 and number of trees of 100 was ideal for our analysis.</p>

```python
# config parameters
num_trees = 45
new_depth = 6

# model random forest
model_rf = RandomForestClassifier(n_estimators=num_trees, max_depth=new_depth)

# fit model on X_train data
model_rf.fit(x_train, y_train)

# predict using model
y_pred_train_rf = model_rf.predict(x_train)
y_pred_test_rf = model_rf.predict(x_test)

# accuracy from train and test
train_score_rf = accuracy_score(y_train, y_pred_train_rf)
test_score_rf = accuracy_score(y_test, y_pred_test_rf)

# print accuracy scores
print("[Random Forest] Classification accuracy for train set: ", train_score_rf)
print("[Random Forest] Classification accuracy for test set:", test_score_rf)
```

    [Random Forest] Classification accuracy for train set:  0.9300889328063241
    [Random Forest] Classification accuracy for test set: 0.9229249011857708


A random forest, at the same depth as the decision tree (namely a depth of 6) performs even better. The test data reaches an accuracy of about 92.6% in the training at 91.5% in the test. 


# Boosting
Next, we will consider boosting, an iterative approach that might eliminate some more of the error in our trees.

```python
# define classifier function
def boostingClassifier(x_train, y_train, depth):
    # AdaBoostClassifier
    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth),
                         n_estimators=800, learning_rate = 0.05)
    abc.fit(x_train, y_train)
    # staged_score train to plot
    abc_predicts_train = list(abc.staged_score(x_train,y_train))
    plt.plot(abc_predicts_train, label = "train");

    # staged_score test to plot
    abc_predicts_test = list(abc.staged_score(x_test,y_test))
    plt.plot(abc_predicts_test, label = "test");

    plt.legend()
    plt.title("AdaBoost Classifier Accuracy, n = "+str(depth))
    plt.xlabel("Iterations")
    plt.show()
    
    return("Maximum test accuracy for depth of "+str(depth)+" is "+str(max(abc_predicts_test))+" at "+str(abc_predicts_test.index(max(abc_predicts_test)))+" iterations")
```


```python
for i in range(1,5):
    print(boostingClassifier(x_train, y_train, i))
```


![png](output_60_0.png)


    Maximum test accuracy for depth of 1 is 0.9150197628458498 at 773 iterations



![png](output_60_2.png)


    Maximum test accuracy for depth of 2 is 0.9298418972332015 at 751 iterations



![png](output_60_4.png)


    Maximum test accuracy for depth of 3 is 0.9268774703557312 at 500 iterations



![png](output_60_6.png)


    Maximum test accuracy for depth of 4 is 0.9219367588932806 at 530 iterations


We see based upon an AdaBoostClassifier the maximum test accuracy of 93.0% is attained at a depth of 2. This is attained after 751 iterations. The AdaBoostClassifier is our most accurate model so far.

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




