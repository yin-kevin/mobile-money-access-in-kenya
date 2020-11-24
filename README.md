# mobile-money-access-in-kenya

![Reach+Alliance+logo](https://user-images.githubusercontent.com/74945619/100048710-77764d80-2de3-11eb-9c6b-8255d914309d.png)

# Introduction

This is a Python coding sample for pre-doctoral applications in economics. The code was written as part of my research with a group of students from the University of Toronto studying mobile money in Kenya and the barriers to last mile financial inclusion. Here I use FSD-Kenya household-level survey data to identify the socioeconomic characteristics that correlate with, and predict, a person's likelihood of *not* being reached by mobile money services. It includes data cleaning, linear regression and regularized logistic regression to predict non-users. 

We were interested in non-users as opposed to users in alignment with the theme of understanding minority populations that are left behind by social interventions. Our research on this topic is part of a broader initiative at the university called the
[Reach Alliance](http://reachalliance.org/) at the [Munk School of Global Affairs and Public Policy](https://munkschool.utoronto.ca/) which seeks to understand the ways we can get valuable services to these 'hardest-to-reach' groups. 


# Contents

The code is divided into cells. Some cells, particularly in the steps tuning the hyperparameters for logistic regression, exist on their own purely because they have long run times. These computationally expensive cells are indicated with comments.

**1.** Imports and Data Cleaning
2. Linear Regression: Raw
3. Linear Regression: 1st Edit
3. Linear Regression: 2nd Edit
4. Logistic Regression: Balance Data, Normalize
4. Logistic Regression: Regularization Parameter
