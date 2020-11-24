# mobile-money-access-in-kenya

![Reach+Alliance+logo](https://user-images.githubusercontent.com/74945619/100048710-77764d80-2de3-11eb-9c6b-8255d914309d.png)

# Introduction

This is a Python coding sample for pre-doctoral applications in economics. The code was written as part of my research with a group of students from the University of Toronto studying mobile money in Kenya and the barriers to last mile financial inclusion. Here I use FSD-Kenya household-level survey data to identify the socioeconomic characteristics that correlate with, and predict, a person's likelihood of *not* being reached by mobile money services. It includes data cleaning, linear regression and regularized logistic regression to predict non-users. 

We were interested in non-users as opposed to users in alignment with the theme of understanding minority populations that are left behind by social interventions. We are particularly interested in the features that are selected and their corresponding coefficients as a rough guide to their importance. Our research on this topic is part of a broader initiative at the university called the
[Reach Alliance](http://reachalliance.org/) at the [Munk School of Global Affairs and Public Policy](https://munkschool.utoronto.ca/) which seeks to understand the ways we can get valuable services to these 'hardest-to-reach' groups. 


# Contents

The code is divided into cells. Some cells, particularly in the steps tuning the hyperparameters for logistic regression, exist on their own purely because they have long run times. These computationally expensive cells are indicated with comments. 

We are not trying to establish causal inference with this kind of data, however we still want to make sure that we aren't ignoring any notable correlations. Thus we start with a large linear regression with as many controls as are feasible with the dataset and eliminate those that are not instructive. The logistic regression is for trying to understand what features help for predictions; it is not being used to describe the real world. Thus we can balance the data by oversampling. We try two feature selection methods just to make sure the results are comparable. 

# Results

We can plot bar graphs of the coefficient size on the categorical variables for linear regression and their statistical significance. The continuous variables are not shown here for brevity. 

![OLS_2_cat_2018](https://user-images.githubusercontent.com/74945619/100051159-33864700-2de9-11eb-9e1f-bb4816906aa3.png)

We can do the same for logistic regression, with the caveat that SciKit Learn does not allow us to extract statistical significance.

![lasso_cat_2018](https://user-images.githubusercontent.com/74945619/100051185-43059000-2de9-11eb-88f7-837115f997f9.png)

Some immediate findings are the clear importance of cellphone ownership and formal identification in determining access, both of which are prerequisites for having a mobile money account. The demographic characteristics are less clear, where the logistic regression drops minority religious and language groups as they are less helpful in predictions on the general population. The final regularized logistic regression model has a 90% accuracy rate when predicting out of sample.
