# mobile-money-access-in-kenya


<p align="center">
  <img src="https://user-images.githubusercontent.com/74945619/100048710-77764d80-2de3-11eb-9c6b-8255d914309d.png" alt="Reach Logo" width="700"/>
</p>


# Introduction

The code here was written as part of my research with a group of students from the University of Toronto studying mobile money in Kenya and the barriers to last mile financial inclusion. Here I use FSD-Kenya household-level survey data to identify the socioeconomic characteristics that correlate with, and predict, a person's likelihood of *not* being reached by mobile money services. It includes data cleaning, linear regression and regularized logistic regression to predict non-users. 

We were interested in non-users as opposed to users in alignment with the theme of understanding minority populations that are **left behind** by social interventions. We are particularly interested in the features that are selected and their corresponding coefficients as a rough guide to their importance. Our research on this topic is part of a broader initiative at the university called the
[Reach Alliance](http://reachalliance.org/) at the [Munk School of Global Affairs and Public Policy](https://munkschool.utoronto.ca/) which seeks to understand the ways we can get valuable services to these 'hardest-to-reach' groups. 


<p align="center">
  <img src="https://user-images.githubusercontent.com/74945619/104352088-77312f00-54c3-11eb-9697-81355fa61d05.png" alt="County Map" width="400"/>
  <img src="https://user-images.githubusercontent.com/74945619/104352471-f6266780-54c3-11eb-8539-033951317ed1.png" alt="Data Map" width="400"/>
</p>
  

# Contents

The code is divided into cells. Some cells, particularly in the steps tuning the hyperparameters for logistic regression, are seperate because they have long run times. These computationally expensive cells are indicated with comments. 

We are not trying to establish causal inference with this kind of data, however we still want to make sure that we aren't ignoring any notable correlations. Thus we start with a large linear regression with as many controls as are feasible with the dataset and eliminate those that are not economically and/or statistically instructive. The logistic regression is for trying to understand what features help for predictions; it is not being used to describe the real world. Thus we can balance the data by oversampling. We try two feature selection methods just to make sure the results are robust. 

# Results

The full results will be in our upcoming published report. As a sample, we can plot bar graphs of the coefficient size on the categorical variables for linear regression and their statistical significance. The continuous variables are not shown here for brevity. We can do the same for logistic regression, with the caveat that SciKit Learn does not allow us to extract statistical significance.

<p align="center">
  <img id="lin_coef" src="https://user-images.githubusercontent.com/74945619/100051159-33864700-2de9-11eb-9e1f-bb4816906aa3.png" alt="Linear Regression Coefficients" width="600"/>
  #lin_coef img {
  margin-left: 500px;
  }

<img id="log_coef" src="https://user-images.githubusercontent.com/74945619/100051185-43059000-2de9-11eb-88f7-837115f997f9.png" alt="Logistic Regression Coefficients" width="730"/>

Some immediate findings are that a lack of cellphone ownership and formal identification are clear barriers, both of which are prerequisites for having a mobile money account. The demographic characteristics are less clear, where the logistic regression drops some minority religious and language groups as they are less helpful in predictions on the general population. The final regularized logistic regression model has a 90% accuracy rate when predicting out of sample.
