# mobile-money-access-in-kenya

![Reach+Alliance+logo](https://user-images.githubusercontent.com/74945619/100048710-77764d80-2de3-11eb-9c6b-8255d914309d.png)

# Introduction

This is a Python coding sample for pre-doctoral applications in economics. The code was written as part of my research with a group of students from the University of Toronto studying mobile money in Kenya and the barriers to last mile financial inclusion. Here I use FSD-Kenya household-level survey data to identify the socioeconomic characteristics that correlate with, and predict, a person's likelihood of *not* being reached by mobile money services. It includes data cleaning, linear regression and regularized logistic regression to predict non-users. 

We were interested in non-users as opposed to users in alignment with the theme of understanding minority populations that are left behind by social interventions. We are particularly interested in the features that are selected and their corresponding coefficients as a rough guide to their importance. Our research on this topic is part of a broader initiative at the university called the
[Reach Alliance](http://reachalliance.org/) at the [Munk School of Global Affairs and Public Policy](https://munkschool.utoronto.ca/) which seeks to understand the ways we can get valuable services to these 'hardest-to-reach' groups. 


# Contents

The code is divided into cells. Some cells, particularly in the steps tuning the hyperparameters for logistic regression, exist on their own purely because they have long run times. These computationally expensive cells are indicated with comments. 

Since we are not trying to establish causal inference, we simply want to make sure that we aren't ignoring any notable correlations. Thus we start with a large linear regression with as many controls as are feasible with the data and eliminate those that are not instructive. The logistic regression is trying to understand what features help for predictions; it is not being used to describe the real world. Thus we can balance the data by oversampling. We try two feature selection methods just to make sure the results are comparable. 

**1.** Imports and Data Cleaning <br/>
**2.** Linear Regression: Raw <br/>
**3.** Linear Regression: 1st Edit <br/>
**3.** Linear Regression: 2nd Edit <br/>
**4.** Logistic Regression: Balance Data, Normalize <br/>
**5.** Logistic Regression: Regularization Parameter <br/>
**6.** Logistic Regression: Learning Curves <br/>
**7.** Logistic Regression: Lasso Selection <br/>
**8.** Logistic Regression: Plot Number of Features Against Accuracy <br/>
**9.** Logistic Regression: Recursive Feature Elimination <br/>
**10.** Logistic Regression: Performance <br/>

# Key Results

We can plot bar graphs of the coefficient size on the categorical variables for linear regression and their statistical significance. The continuous variables are not shown here for brevity. 


We can do the same for logistic regression, with the caveat that SciKit Learn does not allow us to extract statistical significance.


Some immediate findings are the clear importance of cellphone ownership and formal identification in determining access, both of which are prerequisites for having a mobile money account. The demographic characteristics are less clear, where the logistic regression drops minority religious and language groups as they are less helpful in predictions on the general population.
