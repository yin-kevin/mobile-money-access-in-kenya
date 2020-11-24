#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:05:51 2020

@author: kevinyin
"""

# This file contains part of the code I wrote for the Kenya M-Pesa research project mentioned in my CV and cover letter
# It covers data cleaning, manipulation, linear regression and logistic regression models
# We used linear regression to identify correlations of interest and the other ML models to improve our prediction accuracy for non-users
# As it was a mixed-methods interdisciplinary study, we were not attempting to identify causal effects 


# I recommend running each cell individually because some are more computationally expensive than others
# Cells which are computationally more expensive than others have a WARNING comment under the title of the cell
# The 'LOGISTIC REGRESSION: REGULARIZATION PARAMETER' cell in particular has a very long run time




# %% IMPORTS, DATA CLEANING


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import statsmodels.api as sm
import sklearn as sk
from sklearn import feature_selection
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTENC

# Set directory
os.chdir("/Users/kevinyin/Documents/University of Toronto/Reach/Data Analysis")

# Load data and call it "finacc_2018"
finacc_2018 = pd.read_csv("finaccess2018raw.csv")



# *************************
# ***** VARIABLE LIST *****
# *************************

# --- Dependent ---

# M-MONEY USE - c1a9
# M-MONEY ACCESS - mobile_money_access

# --- Independent --- 

# URBAN/RURAL - cluster_type
# COUNTY - a1
# HOUSEHOLD SIZE - a10
# AGE - a13
# GENDER - gender
# LANGUAGE - a15
# RELIGION - y3
# MARITAL STATUS - a17
# EDUCATION - education
# IDENTIFICATION - y19_a, y19_b, y19_c
# MONTHLY INCOME - b3h1
# M-PHONE OWNERSHIP - mobile
# M-PHONE ACCESS - Mobile_access
# NEAREST FIN-SERVICE - x1
# PROPERTY OWNERSHIP - y15

# --- Other ---

# M-MONEY USE CASE - k1i
# WEALTH QUINTILE - wealth_quint
# INCOME GROUP - incomegp
# H-HEAD EDUCATION - a19
# FEMALE H-HEAD EDUCATION - a18
# VULNERABILITIES (FOOD ETC.) - b1c1_1, b1c1_2, b1c1_3, b1c1_4
# FINANCIAL LITERACY - b2g
# FINANCIAL COMPREHENSION - b2h
# BORROWED M-MONEY ACCOUNT USE - k1a
# PERSONAL M-MONEY ACCOUNT ONLY - k1c_1
# SHARED M-MONEY ACCOUNT - k1c_2 
# M-MONEY PROVIDER(S) - k1d_1 (primary), k1d_2 (secondary), k1d_3 (tertiary)
# BIGGEST M-MONEY CHALLENGE - k1n
# TRANSPORT COST TO M-MONEY AGENT - x6
# WALK TIME TO M-MONEY AGENT - x7
# VULNERABILITY INDEX - vul_index

# ***** End *****



# *************************
# ***** BAD RESPONSES *****
# *************************

# Counting 'refusals to answer', 'don't know' and other odd values for various predictors
print()
print('Responded "Refused to Answer" to Religion:', finacc_2018.y3.eq('Refused to Answer').sum()/finacc_2018.shape[0])
print("Responded 'Don't know' to Religion:", finacc_2018.y3.eq("Don't know").sum()/finacc_2018.shape[0])
print('Responded "Refused to Answer" to Property Ownership:', finacc_2018.y15.eq('Refused to Answer').sum()/finacc_2018.shape[0])
print('Responded "Refused to Answer" to Nearest Finservice:', finacc_2018.x1.eq('Refused to answer').sum()/finacc_2018.shape[0])
print('Responded "Don?t know" to Nearest Finservice:', finacc_2018.x1.eq('Don?t know').sum()/finacc_2018.shape[0])
print('Responded "Refused to Answer" to Marital Status:', finacc_2018.a17.eq('Refused to Answer').sum()/finacc_2018.shape[0])
print("Responded 'Don't know' to Marital Status:", finacc_2018.a17.eq("Don't know").sum()/finacc_2018.shape[0])
print('Responded "Refused to Answer" to Mobile Access:', finacc_2018.mobile_access.eq('Refused to answer').sum()/finacc_2018.shape[0])
print()
# Most of these show less than 0.1% of respondents chose these responses so we can drop them
# Except for nearest financial service, where 2.4% of respond said they don't know
# Total 228 dropped observations

# Drop observations where people refused to respond
finacc_2018 = finacc_2018[(finacc_2018['y3'] != 'Refused to Answer') & (finacc_2018['a17'] != 'Refused to Answer') & 
                          (finacc_2018['y15'] != 'Refused to answer') &
                          (finacc_2018['x1'] != 'Refused to answer') & 
                          (finacc_2018['x1'] != 'Don?t know') & (finacc_2018['y3'] != "Don't know") &
                          (finacc_2018['a17'] != "Don't know") & (finacc_2018['mobile_access'] != 'Refused to answer')]



# ***************************
# ***** DUMMY VARIABLES *****
# ***************************


# Creating necessary dummy variables (except for identification) 
# Did not use county dummies because we used population density instead
cluster_dum = pd.get_dummies(finacc_2018['cluster_type']).rename(columns=lambda x: 'cluster_' + str(x))
county_dum = pd.get_dummies(finacc_2018['a1']).rename(columns=lambda x: 'county_' + str(x))
gender_dum = pd.get_dummies(finacc_2018['gender']).rename(columns=lambda x: 'gender_' + str(x))
language_dum = pd.get_dummies(finacc_2018['a15']).rename(columns=lambda x: 'language_' + str(x))
religion_dum = pd.get_dummies(finacc_2018['y3']).rename(columns=lambda x: 'religion_' + str(x))
marital_dum = pd.get_dummies(finacc_2018['a17']).rename(columns=lambda x: 'marital_status_' + str(x))
education_dum = pd.get_dummies(finacc_2018['education']).rename(columns=lambda x: 'education_' + str(x))
mobile_access_dum = pd.get_dummies(finacc_2018['mobile_access']).rename(columns=lambda x: 'mobile_access_' + str(x))
nearest_finservice_dum = pd.get_dummies(finacc_2018['x1']).rename(columns=lambda x: 'nearest_finservice_' + str(x))
property_owner_dum = pd.get_dummies(finacc_2018['y15']).rename(columns=lambda x: 'property_owner_' + str(x))


# Creating identification-category dummies (if we want to decompose the types of ID)
national_id_dum = pd.get_dummies(finacc_2018['y19_a']).rename(columns=lambda x: 'national_id_' + str(x))
passport_dum = pd.get_dummies(finacc_2018['y19_b']).rename(columns=lambda x: 'passport_' + str(x))
foreign_id_dum = pd.get_dummies(finacc_2018['y19_c']).rename(columns=lambda x: 'foreign_id_' + str(x))


# Creating binary-identification dummy
def test_func_1(finacc_2018):
    if finacc_2018['y19_a'] == 'Yes' or finacc_2018['y19_b'] == 'Yes' or finacc_2018['y19_c'] == 'Yes':
        return 0
    else:
        return 1
no_identification = ((finacc_2018.apply(test_func_1, axis=1)).to_frame()).rename(columns=lambda x: 'no_identification')


# Creating no_phone dummy (flipping mobile_ownership)
def test_func_1(finacc_2018):
    if finacc_2018['mobile'] == 1:
        return 0
    else:
        return 1
no_phone = ((finacc_2018.apply(test_func_1, axis=1)).to_frame()).rename(columns=lambda x: 'no_phone')


# Creating binary mobile money use dummy ('unreached' means non-user of mobile money as they are unreached by the service)
def test_func_2(finacc_2018):
    if finacc_2018['c1a9'] == 'Currently Use' or finacc_2018['c1a9'] == 'Used to Use':
        return 0
    else:
        return 1
unreached = ((finacc_2018.apply(test_func_2, axis=1)).to_frame()).rename(columns=lambda x: 'unreached')


# Creating alternative education binary dummies (as opposed to a single multi-categorical) for each threshold

# No education
def test_func_3(finacc_2018):
    if finacc_2018['education'] == 'None':
        return 1
    else:
        return 0
no_education = ((finacc_2018.apply(test_func_3, axis=1)).to_frame()).rename(columns=lambda x: 'no_education')

# At least primary education
def test_func_4(finacc_2018):
    if finacc_2018['education'] == 'Primary' or finacc_2018['education'] == 'Secondary' or finacc_2018['education'] == 'Tertiary':
        return 1
    else:
        return 0
primary_ed = ((finacc_2018.apply(test_func_4, axis=1)).to_frame()).rename(columns=lambda x: 'primary_ed')

# At least secondary education (high school)
def test_func_5(finacc_2018):
    if finacc_2018['education'] == 'Secondary' or finacc_2018['education'] == 'Tertiary':
        return 1
    else:
        return 0
secondary_ed = ((finacc_2018.apply(test_func_5, axis=1)).to_frame()).rename(columns=lambda x: 'secondary_ed')

# At least tertiary education (college, university, graduate training)
def test_func_6(finacc_2018):
    if finacc_2018['education'] == 'Tertiary':
        return 1
    else:
        return 0
tertiary_ed = ((finacc_2018.apply(test_func_6, axis=1)).to_frame()).rename(columns=lambda x: 'tertiary_ed')


# Concatenating dummy dataframes
finacc_2018_with_dummies = pd.concat([unreached, cluster_dum, county_dum, gender_dum, language_dum, 
                                      religion_dum, marital_dum, primary_ed, secondary_ed, tertiary_ed,
                                      nearest_finservice_dum, property_owner_dum, no_education, no_identification, no_phone], axis=1)
finacc_2018_with_dummies.rename(columns={'nearest_finservice_Mobile Money Agent (for depositing or withdrawing cash)':'nearest_finservice_Mobile Money Agent',
                                         'property_owner_No':'no_property', 
                                         'property_owner_Yes':'yes_property'}, inplace=True)


# Creating dataframe for non-dummy variables and concatenating it with dummy dataframe
finacc_2018_not_dummies = finacc_2018[['a1', 'a10', 'a13', 'b3h1']]
finacc_2018_not_dummies.rename(columns={'a10':'household_size', 'a13':'age', 'b3h1':'monthly_income'}, inplace=True)
final_finacc_pre_drop = pd.concat([finacc_2018_with_dummies, finacc_2018_not_dummies], axis=1)
                             

# Checking percentage of dataset that uses mobile money, shows 76%
print(final_finacc_pre_drop.unreached.eq(1).sum()/finacc_2018.shape[0])



# ***************************************
# ***** ADDING POPULATION DENSITIES *****
# ***************************************


# Adding population densities by county (using data from the Kenyan Statistics Bureau 2019 Household Survey)
conditions = [(final_finacc_pre_drop['a1'] == 'Mombasa'), 
              (final_finacc_pre_drop['a1'] == 'Kwale'),
              (final_finacc_pre_drop['a1'] == 'Kilifi'),
              (final_finacc_pre_drop['a1'] == 'Tana River'),
              (final_finacc_pre_drop['a1'] == 'Lamu'),
              (final_finacc_pre_drop['a1'] == 'Taita Taveta'),
              (final_finacc_pre_drop['a1'] == 'Garissa'), 
              (final_finacc_pre_drop['a1'] == 'Wajir'),
              (final_finacc_pre_drop['a1'] == 'Mandera'),
              (final_finacc_pre_drop['a1'] == 'Marsabit'),
              (final_finacc_pre_drop['a1'] == 'Isiolo'),
              (final_finacc_pre_drop['a1'] == 'Meru'), 
              (final_finacc_pre_drop['a1'] == 'Tharaka'),
              (final_finacc_pre_drop['a1'] == 'Embu'),
              (final_finacc_pre_drop['a1'] == 'Kitui'),
              (final_finacc_pre_drop['a1'] == 'Machakos'),
              (final_finacc_pre_drop['a1'] == 'Makueni'),
              (final_finacc_pre_drop['a1'] == 'Nyandarua'), 
              (final_finacc_pre_drop['a1'] == 'Nyeri'),
              (final_finacc_pre_drop['a1'] == 'Kirinyaga'),
              (final_finacc_pre_drop['a1'] == 'Muranga'),
              (final_finacc_pre_drop['a1'] == 'Kiambu'),
              (final_finacc_pre_drop['a1'] == 'Turkana'), 
              (final_finacc_pre_drop['a1'] == 'West Pokot'),
              (final_finacc_pre_drop['a1'] == 'Samburu'),
              (final_finacc_pre_drop['a1'] == 'Trans-Nzoia'),
              (final_finacc_pre_drop['a1'] == 'Uasin Gishu'),
              (final_finacc_pre_drop['a1'] == 'Elgeyo Marakwet'), 
              (final_finacc_pre_drop['a1'] == 'Nandi'),
              (final_finacc_pre_drop['a1'] == 'Baringo'),
              (final_finacc_pre_drop['a1'] == 'Laikipia'),
              (final_finacc_pre_drop['a1'] == 'Nakuru'),
              (final_finacc_pre_drop['a1'] == 'Narok'), 
              (final_finacc_pre_drop['a1'] == 'Kajiado'),
              (final_finacc_pre_drop['a1'] == 'Kericho'),
              (final_finacc_pre_drop['a1'] == 'Bomet'),
              (final_finacc_pre_drop['a1'] == 'Kakamega'),
              (final_finacc_pre_drop['a1'] == 'Vihiga'),
              (final_finacc_pre_drop['a1'] == 'Bungoma'),
              (final_finacc_pre_drop['a1'] == 'Busia'),
              (final_finacc_pre_drop['a1'] == 'Siaya'), 
              (final_finacc_pre_drop['a1'] == 'Kisumu'),
              (final_finacc_pre_drop['a1'] == 'Homa Bay'),
              (final_finacc_pre_drop['a1'] == 'Migori'), 
              (final_finacc_pre_drop['a1'] == 'Kisii'),
              (final_finacc_pre_drop['a1'] == 'Nyamira'),
              (final_finacc_pre_drop['a1'] == 'Nairobi')
    ]

    
densities = [5495, 105, 116, 8, 23, 20, 19, 14, 33, 6, 
             11, 221, 153, 216, 37, 235, 121, 194, 228, 413, 
             419, 952, 14, 68, 15, 397, 343, 150, 310, 61, 
             54, 290, 65, 51, 370, 346, 618, 1047, 552, 527, 
             393, 554, 359, 427, 958, 675, 6247]

final_finacc_pre_drop['pop_density'] = np.select(conditions, densities)


# Dropping observations where "Monthly Income" has empty cells
Final_Finacc = final_finacc_pre_drop.dropna(how='any', subset=['monthly_income'])


# Dropping county column
Final_Finacc = Final_Finacc.drop(columns='a1')





# %% LINEAR REGRESSION: RAW


# Creating dataframe for linear regression (this step will be almost identical to log regression)
Final_Finacc_lin = Final_Finacc.drop(columns=list(Final_Finacc.filter(regex='county_')))


# Dropping one variable from each categorical to determine reference group
Final_Finacc_lin = Final_Finacc_lin.drop(['gender_Male', 
                                          'language_Swahili', 
                                          'religion_Christianity',
                                          'cluster_Urban',
                                          'marital_status_Single/Never Married',
                                          'yes_property'
                                          ], axis=1)


# Creating separate dataframes where each one uses a different education dummy
df_linr_some = Final_Finacc_lin.drop(['primary_ed', 'secondary_ed', 'tertiary_ed'], axis=1)
df_linr_primary = Final_Finacc_lin.drop(['no_education', 'secondary_ed', 'tertiary_ed'], axis=1)
df_linr_secondary = Final_Finacc_lin.drop(['primary_ed', 'no_education', 'tertiary_ed'], axis=1)
df_linr_tertiary = Final_Finacc_lin.drop(['primary_ed', 'secondary_ed', 'no_education'], axis=1)


# Setting feature columns to every column except the first (indexed starting at 0)
lin_reg_features_some = df_linr_some.iloc[:,1:]
lin_reg_features_primary = df_linr_primary.iloc[:,1:]
lin_reg_features_secondary = df_linr_secondary.iloc[:,1:]
lin_reg_features_tertiary = df_linr_tertiary.iloc[:,1:]


# Setting matrices for features and dependent variables, trying a regression for each education dummy
# Commented out because we did not end up choosing the other education thresholds

# No education
X1 = lin_reg_features_some
X1 = sm.add_constant(X1)
y1 = df_linr_some.unreached
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.25,random_state=2)
linreg = sm.OLS(y1_train, X1_train).fit()
predictions = linreg.predict()
print()
print(linreg.summary())
print()

# Primary education
# =============================================================================
# X1 = lin_reg_features_primary
# X1 = sm.add_constant(X1)
# y1 = df_linr_primary.unreached
# X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.25,random_state=2)
# linreg = sm.OLS(y1_train, X1_train).fit()
# predictions = linreg.predict()
# print(linreg.summary())
# =============================================================================

# Secondary education
# =============================================================================
# X1 = lin_reg_features_secondary
# X1 = sm.add_constant(X1)
# y1 = df_linr_secondary.unreached
# X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.25,random_state=2)
# linreg = sm.OLS(y1_train, X1_train).fit()
# predictions = linreg.predict()
# print(linreg.summary())
# =============================================================================

# Tertiary education
# =============================================================================
# X1 = lin_reg_features_tertiary
# X1 = sm.add_constant(X1)
# y1 = df_linr_tertiary.unreached
# X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.25,random_state=2)
# linreg = sm.OLS(y1_train, X1_train).fit()
# predictions = linreg.predict()
# print(linreg.summary())
# =============================================================================


# Printing OLS Regression into image (for draft report)
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(linreg.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.show()





# %% LINEAR REGRESSION: 1st EDIT


# Removing statistically insignificant features: [urban/rural, marital status, household size]
Final_Finacc_lin = df_linr_some.drop(columns=['marital_status_Divorced/separated', 
                                              'marital_status_Married/Living with partner',
                                              'marital_status_Widowed', 
                                              'cluster_Rural',
                                              'household_size',
                                              'no_property',
                                                   ])


# Getting feature columns
lin_reg_features2 = Final_Finacc_lin.iloc[:,1:]


# Running regression
X1 = lin_reg_features2
X1 = sm.add_constant(X1)
y1 = Final_Finacc_lin.unreached
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.25,random_state=2)
linreg2 = sm.OLS(y1_train, X1_train).fit()
print()
print(linreg2.summary())
print()


# Comparing predictions to actual training data
probabilities = linreg2.predict(X1_test)
def classifier(y):
    if y >= 0.5:
        return 1
    else:
        return 0
y1_pred = (probabilities.apply(classifier)).to_frame()


# Printing accuracy and classification report
print()
print('Linear Regression Classification Accuracy: {}.'.format(sk.metrics.accuracy_score(y1_test,y1_pred)))
print()
print(sk.metrics.classification_report(y1_test, y1_pred))
print()


# Printing OLS Regression into image
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(linreg2.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.show()


# Printing confusion matrix
cm2 = sk.metrics.confusion_matrix(y1_test, y1_pred)
ax = plt.subplot()
sns.heatmap(cm2, annot=True, fmt='g', cmap='Blues', ax = ax) 
ax.set_xlabel('Predicted Mobile Money Use');ax.set_ylabel('True Mobile Money Use')
ax.set_title('Linear Regression Confusion Matrix #1 (2018)') 
plt.show()


# Creating categorical coefficients dataframe and ordering for visualization
lin_categorical = linreg2.params.drop(['age','monthly_income','pop_density'])
cat_pvalues = linreg2.pvalues.drop(['age','monthly_income','pop_density'])

df_lin_coef = pd.DataFrame(lin_categorical)
df_lin_abs = pd.DataFrame(map(abs, df_lin_coef[0]), index=df_lin_coef.index) 
df_lin_coef = pd.concat([df_lin_coef,df_lin_abs,cat_pvalues], axis=1)
df_lin_coef.columns = ['coefficient', 'abs_value', 'p_value']
df_lin_coef = df_lin_coef.sort_values(by=['abs_value'], ascending=True)

            
# Setting color mask, red if positive, blue if negative, darker if statistically significant
color_mask  = []
for c,p in zip(df_lin_coef['coefficient'], df_lin_coef['p_value']):
        if c < 0:
            if p < 0.05:
                color_mask.append('royalblue')
            elif p > 0.05:
                color_mask.append('lightskyblue')
        if c > 0:
            if p < 0.05:
                color_mask.append('crimson')
            elif p > 0.05: 
                color_mask.append('lightpink')


# Plotting feature coefficients
y_pos = np.arange(len(df_lin_coef.index))
plt.barh(y_pos, df_lin_coef['abs_value'], color=color_mask)
plt.yticks(y_pos, df_lin_coef.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach (ss)')
red_patch_2 = mpatches.Patch(color='lightpink', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach (ss)')
blue_patch_2 = mpatches.Patch(color='lightskyblue', label='Easier to Reach')
plt.legend(handles=[red_patch, red_patch_2, blue_patch, blue_patch_2])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("Categorical Linear Regression #1 Coefficients (2018)")
plt.show()


# Creating continuous coefficients dataframe and ordering for visualization
lin_continuous = linreg2.params.filter(['age','monthly_income','pop_density'])
cont_pvalues = linreg2.pvalues.filter(['age','monthly_income','pop_density'])

df_lin_coef = pd.DataFrame(lin_continuous)
df_lin_abs = pd.DataFrame(map(abs, df_lin_coef[0]), index=df_lin_coef.index) 
df_lin_coef = pd.concat([df_lin_coef,df_lin_abs, cont_pvalues], axis=1)
df_lin_coef.columns = ['coefficient', 'abs_value', 'p_value']
df_lin_coef = df_lin_coef.sort_values(by=['abs_value'], ascending=True)


# Setting color mask, red if positive, blue if negative, darker if statistically significant
color_mask  = []
for c,p in zip(df_lin_coef['coefficient'], df_lin_coef['p_value']):
        if c < 0:
            if p < 0.05:
                color_mask.append('royalblue')
            elif p > 0.05:
                color_mask.append('lightskyblue')
        if c > 0:
            if p < 0.05:
                color_mask.append('crimson')
            elif p > 0.05: 
                color_mask.append('lightpink')


# Bar chart of continuous coefficient values
y_pos = np.arange(len(df_lin_coef.index))
plt.barh(y_pos, df_lin_coef['abs_value'], color=color_mask)
plt.yticks(y_pos, df_lin_coef.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach (ss)')
red_patch_2 = mpatches.Patch(color='lightpink', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach (ss)')
blue_patch_2 = mpatches.Patch(color='lightskyblue', label='Easier to Reach')
plt.legend(handles=[red_patch, red_patch_2, blue_patch, blue_patch_2])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("Continuous Linear Regression #1 Coefficients (2018)")
plt.show()





# %% LINEAR REGRESSION: 2nd EDIT


# Removing nearest financial service features in addition to the features removed in regression 2
Final_Finacc_lin = df_linr_some.drop(columns=['nearest_finservice_Bank/Post Bank Branch/ Head Office/ ATM',
                                              'nearest_finservice_Microfinance Institution', 
                                              'nearest_finservice_Sacco',
                                              'marital_status_Divorced/separated', 
                                              'marital_status_Married/Living with partner',
                                              'marital_status_Widowed', 
                                              'nearest_finservice_Bank/Post Bank Branch/ Head Office/ ATM',
                                              'nearest_finservice_Microfinance Institution', 
                                              'nearest_finservice_Sacco',
                                              'nearest_finservice_Bank Agent/ Post Bank Agent',
                                              'nearest_finservice_Mobile Money Agent',
                                              'cluster_Rural',
                                              'household_size',
                                              'no_property',
                                                   ])


# Getting feature columns
lin_reg_features3 = Final_Finacc_lin.iloc[:,1:]


# Running regression
X1 = lin_reg_features3
X1 = sm.add_constant(X1)
y1 = Final_Finacc_lin.unreached
X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.25,random_state=2)
linreg3 = sm.OLS(y1_train, X1_train).fit()
print()
print(linreg2.summary())
print()

# Comparing predictions to actual training data
probabilities = linreg3.predict(X1_test)
def classifier(y):
    if y >= 0.5:
        return 1
    else:
        return 0
y1_pred = (probabilities.apply(classifier)).to_frame()


# Printing accuracy and classification report
print()
print('Linear Regression Classification Accuracy: {}.'.format(sk.metrics.accuracy_score(y1_test,y1_pred)))
print()
print(sk.metrics.classification_report(y1_test, y1_pred))
print()


# Printing OLS Regression into image
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(linreg3.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()
plt.show()


# Printing confusion matrix
cm3 = sk.metrics.confusion_matrix(y1_test, y1_pred)
ax = plt.subplot()
sns.heatmap(cm3, annot=True, fmt='g', cmap='Blues', ax = ax) 
ax.set_xlabel('Predicted Mobile Money Use');ax.set_ylabel('True Mobile Money Use')
ax.set_title('Linear Regression Confusion Matrix #2 (2018)') 
plt.show()


# Creating categorical coefficients dataframe and ordering for visualization
lin_categorical = linreg3.params.drop(['age','monthly_income','pop_density'])
cat_pvalues = linreg3.pvalues.drop(['age','monthly_income','pop_density'])

df_lin_coef = pd.DataFrame(lin_categorical)
df_lin_abs = pd.DataFrame(map(abs, df_lin_coef[0]), index=df_lin_coef.index) 
df_lin_coef = pd.concat([df_lin_coef,df_lin_abs,cat_pvalues], axis=1)
df_lin_coef.columns = ['coefficient', 'abs_value', 'p_value']
df_lin_coef = df_lin_coef.sort_values(by=['abs_value'], ascending=True)


# Setting color mask, red if positive, blue if negative, darker if statistically significant
color_mask  = []
for c,p in zip(df_lin_coef['coefficient'], df_lin_coef['p_value']):
        if c < 0:
            if p < 0.05:
                color_mask.append('royalblue')
            elif p > 0.05:
                color_mask.append('lightskyblue')
        if c > 0:
            if p < 0.05:
                color_mask.append('crimson')
            elif p > 0.05: 
                color_mask.append('lightpink')


# Plotting feature coefficients
y_pos = np.arange(len(df_lin_coef.index))
plt.barh(y_pos, df_lin_coef['abs_value'], color=color_mask)
plt.yticks(y_pos, df_lin_coef.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach (ss)')
red_patch_2 = mpatches.Patch(color='lightpink', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach (ss)')
blue_patch_2 = mpatches.Patch(color='lightskyblue', label='Easier to Reach')
plt.legend(handles=[red_patch, red_patch_2, blue_patch, blue_patch_2])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("Categorical Linear Regression #2 Coefficients (2018)")
plt.show()


# Creating continuous coefficients dataframe and ordering for visualization
lin_continuous = linreg2.params.filter(['age','monthly_income','pop_density'])
cont_pvalues = linreg2.pvalues.filter(['age','monthly_income','pop_density'])

df_lin_coef = pd.DataFrame(lin_continuous)
df_lin_abs = pd.DataFrame(map(abs, df_lin_coef[0]), index=df_lin_coef.index) 
df_lin_coef = pd.concat([df_lin_coef,df_lin_abs, cont_pvalues], axis=1)
df_lin_coef.columns = ['coefficient', 'abs_value', 'p_value']
df_lin_coef = df_lin_coef.sort_values(by=['abs_value'], ascending=True)


# Setting color mask, red if positive, blue if negative, darker if statistically significant
color_mask  = []
for c,p in zip(df_lin_coef['coefficient'], df_lin_coef['p_value']):
        if c < 0:
            if p < 0.05:
                color_mask.append('royalblue')
            elif p > 0.05:
                color_mask.append('lightskyblue')
        if c > 0:
            if p < 0.05:
                color_mask.append('crimson')
            elif p > 0.05: 
                color_mask.append('lightpink')


# Bar chart of continuous coefficient values
y_pos = np.arange(len(df_lin_coef.index))
plt.barh(y_pos, df_lin_coef['abs_value'], color=color_mask)
plt.yticks(y_pos, df_lin_coef.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach (ss)')
red_patch_2 = mpatches.Patch(color='lightpink', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach (ss)')
blue_patch_2 = mpatches.Patch(color='lightskyblue', label='Easier to Reach')
plt.legend(handles=[red_patch, red_patch_2, blue_patch, blue_patch_2])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("Continuous Linear Regression #2 Coefficients (2018)")
plt.show()





# %% LOGISTIC REGRESSION: BALANCE DATA, NORMALIZE


# Using 'no_education' as education dummy, removing county dummies
df_log = Final_Finacc.drop(columns=list(Final_Finacc.filter(regex='county_')))
df_log = df_log.drop(['primary_ed', 
                      'secondary_ed', 
                      'tertiary_ed',
                      'cluster_Urban',
                      'gender_Male',
                      'yes_property'
                      ], axis=1)


# Normalizing continuous variables
def normalize(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df

df_log[['household_size','age','monthly_income','pop_density']] = normalize(df_log[['household_size', 'age','monthly_income','pop_density']])


# Setting features and outcome variable
X2 = df_log.drop(columns='unreached')
y2 = df_log.unreached
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.25, random_state=1)


# Collecting categorical columns and getting a list of indices for them
cat_cols = X2.drop(['household_size', 'age', 'monthly_income', 'pop_density'], axis=1)
cat_cols_ind = []
for key in cat_cols:
    ind = X2.columns.get_loc(key)
    cat_cols_ind.append(ind)
    

# Creating function to extract the class distribution of the training examples
def get_class_dist(data, name):
    unique, counts = np.unique(data, return_counts=True)
    pct = 100*(counts/len(data))
    d = dict(zip(unique, zip(counts, pct)))
    print(len(data), 'total examples in %s' % name)
    for key, values in d.items():
        print('class %d: %d examples,' % (key, values[0]), "{0:.2f}%".format(values[1]))
    print('')
    return


# Printing the class distribution of training examples before balancing
print()
print(get_class_dist(data=y2_train, name='y2_train'))
print()

# Applying SMOTENC to balance the data
smote_nc = SMOTENC(categorical_features=cat_cols_ind,random_state = 42)
X2_train, y2_train = smote_nc.fit_resample(X2_train, y2_train)


# Printing the class distribution of training examples after balancing
print()
print(np.bincount(y2_train))
print(get_class_dist(data=y2_train, name='y2_train'))
print()


# Count features
print()
print('Total Number of Features:', len(X2.columns))
print()





# %% LOGISTIC REGRESSION: REGULARIZATION PARAMETER
# WARNING: VERY VERY Computationally Expensive


# Creating a vector of regularization parameters to test
lam = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 1, 3, 10, 30, 100]


# Setting empty lists to fill with mean scores
mean_train_mse = []
mean_test_mse = []
mean_train_acc = []
mean_test_acc = []


# Calculating cross-validated mean squared error and accuracy for different reg-parameters using Lasso
for i in range(len(lam)):
  model = LogisticRegression(penalty='l1', C=(1/lam[i]),  solver='liblinear', max_iter=100000)
  # Note that these lists are the mean scores, not each individual score
  mse = sk.model_selection.cross_validate(model, X2_train, y2_train, scoring='neg_mean_squared_error', 
                                          return_train_score=True, cv=10)
  
  mean_train_mse.append(np.mean(mse['train_score']))
  mean_test_mse.append(np.mean(mse['test_score']))

  accuracy = sk.model_selection.cross_validate(model, X2_train, y2_train, scoring='accuracy', 
                                          return_train_score=True, cv=10)
  mean_train_acc.append(np.mean(accuracy['train_score']))
  mean_test_acc.append(np.mean(accuracy['test_score']))


# Plotting negative mean squared error against logarithm of reg-parameter
plt.plot(np.log(lam), mean_train_mse, label='Training MSE')
plt.plot(np.log(lam), mean_test_mse, label='Test MSE')
plt.xlabel('Log Regularization Parameter')
plt.ylabel('Mean Squared Error')
plt.title('MSE against L1 Regularization Parameter (2018)')
plt.legend()
plt.show()


# Plotting accuracy against logarithm of reg-parameter
plt.plot(np.log(lam), mean_train_acc , label='Training Accuracy')
plt.plot(np.log(lam), mean_test_acc, label='Test Accuracy')
plt.xlabel('Log Regularization Parameter')
plt.ylabel('Accuracy')
plt.title('Accuracy against L1 Regularization Parameter (2018)')
plt.legend()
plt.show()


# Choose a parameter of lambda = 10

 




# %% LOGISTIC REGRESSION: LEARNING CURVES
# WARNING: Computationally Expensive


# Creating vector for different training sizes
n = [100,500,1000,1500,2000,2500,3000,3500,4000,4500]


# Plotting learning curves
train_sizes,train_scores,validation_scores = sk.model_selection.learning_curve(
estimator = LogisticRegression(max_iter=10000),
X = X2_train,
y = y2_train,
train_sizes = n, 
cv = 20,
scoring = 'accuracy',
shuffle = True
)

mean_train_score2 = train_scores.mean(axis=1)
mean_test_score2 = validation_scores.mean(axis=1)

plt.plot(n, mean_train_score2, label='Training Error')
plt.plot(n, mean_test_score2, label='Test Error')
plt.xlabel('Number of Training Examples')
plt.ylabel('Model Accuracy')
plt.title('Learning Curves for Logit Regression Model (2018)')
plt.legend()
plt.show()






# %% LOGISTIC REGRESSION: LASSO SELECTION


# Setting the model
model = LogisticRegression(C=1/10, penalty='l1', solver='liblinear', max_iter=10000)


# Selecting features by eliminating any where regularization sends coefficients to zero
selection = sk.feature_selection.SelectFromModel(model)
selection.fit(X2_train, y2_train)
selected_features1 = X2_train.columns[(selection.get_support())]


# Dataframe of coefficients and ordering them for visuals
coefficients = (selection.estimator_.coef_).ravel().tolist()
coefficients = [x for x in coefficients if x != 0]
abs_coefficients = list(map(abs, coefficients)) 
df_coef = pd.DataFrame(coefficients).set_index(selected_features1)
df_abs = pd.DataFrame(abs_coefficients).set_index(selected_features1)
df_coef = pd.concat([df_coef,df_abs], axis=1)
df_coef.columns = ['coefficient', 'abs_value']
df_coef = df_coef.sort_values(by=['abs_value'], ascending=True)


# Creating seperate dataframes for continuous and categorical features
df_coef_cat = df_coef.drop(['monthly_income', 'pop_density'])
df_coef_cont = df_coef.drop(df_coef_cat.index)


# Creating categorical feature bar chart with red for negative coefficients
color_mask = ['royalblue' if c < 0 else 'crimson' for c in df_coef_cat['coefficient']]
y_pos = np.arange(df_coef_cat.shape[0])
plt.barh(y_pos, df_coef_cat['abs_value'], color=color_mask)
plt.yticks(y_pos, df_coef_cat.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach')
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("Categorical Lasso Logit Features and Coefficients (2018)")
plt.show()


# Creating continuous feature bar chart with red for negative coefficients
color_mask = ['royalblue' if c < 0 else 'crimson' for c in df_coef_cont['coefficient']]
y_pos = np.arange(df_coef_cont.shape[0])
plt.barh(y_pos, df_coef_cont['abs_value'], color=color_mask)
plt.yticks(y_pos, df_coef_cont.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach')
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("Continuous Lasso Logit Features and Coefficients (2018)")
plt.show()





# %% LOGISTIC REGRESSION: PLOT NUMBER OF FEATURES AGAINST PERFORMANCE
# WARNING: Computationally Expensive


# Plotting the number of features against cross-validated accuracies for each model
rfe_selection = sk.feature_selection.RFECV(estimator=model, step=1, cv=10,
              scoring='accuracy')
rfe_selection.fit(X2_train, y2_train)

print()
print('Optimal Number of Features by RFE:', rfe_selection.n_features_)
print()

plt.plot(range(1, len(rfe_selection.grid_scores_) + 1), rfe_selection.grid_scores_)
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.title("RFE Accuracy against Number of Features (2018)")
plt.show()





# %% LOGISTIC REGRESSION: RECURSIVE FEATURE ELIMINATION


# Performing RFE to find the top features
rfe_selection = sk.feature_selection.RFE(estimator=model, step=1)
rfe_selection.fit(X2_train, y2_train)
selected_features2 = X2_train.columns[(rfe_selection.get_support())]


# Creating coefficients dataframe and ordering for visualization
coefficients2 = (rfe_selection.estimator_.coef_).ravel().tolist()
abs_coefficients2 = list(map(abs, coefficients2)) 
df_coef2 = pd.DataFrame(coefficients2).set_index(selected_features2)
df_abs2 = pd.DataFrame(abs_coefficients2).set_index(selected_features2)
df_coef2 = pd.concat([df_coef2,df_abs2], axis=1)
df_coef2.columns = ['coefficient', 'abs_value']
df_coef2 = df_coef2.sort_values(by=['abs_value'], ascending=True)


# Creating seperate dataframes for continuous and categorical features (hardcoded based on selected features)
df_coef2_cat = df_coef2.drop(['monthly_income', 'pop_density'])
df_coef2_cont = df_coef2.drop(df_coef2_cat.index)


# Bar chart of categorical coefficient values
color_mask = ['royalblue' if c < 0 else 'crimson' for c in df_coef2_cat['coefficient']]
y_pos = np.arange(df_coef2_cat.shape[0])
plt.barh(y_pos, df_coef2_cat['abs_value'], color=color_mask)
plt.yticks(y_pos, df_coef2_cat.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach')
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("Categorical RFE Top Features and Coefficients (2018)")
plt.show()


# Bar chart of continuous coefficient values
color_mask = ['royalblue' if c < 0 else 'crimson' for c in df_coef2_cat['coefficient']]
y_pos = np.arange(df_coef2_cont.shape[0])
plt.barh(y_pos, df_coef2_cont['abs_value'], color=color_mask)
plt.yticks(y_pos, df_coef2_cont.index)
red_patch = mpatches.Patch(color='crimson', label='Harder to Reach')
blue_patch = mpatches.Patch(color='royalblue', label='Easier to Reach')
plt.legend(handles=[red_patch, blue_patch])
plt.xlabel("Coefficients")
plt.ylabel("Feature")
plt.title("continuous RFE Top Features and Coefficients (2018)")
plt.show()





# %% LOGISTIC REGRESSION: PERFORMACE


# Model 1: Raw, no feature selection
logreg_raw = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10000)
logreg_raw.fit(X2_train,y2_train)
y2_pred=logreg_raw.predict(X2_test)


# Printing scores and classification report
print()
print("Raw Logistic Regression Scores:")
print("Accuracy:", sk.metrics.accuracy_score(y2_test, y2_pred))
print("Precision:", sk.metrics.precision_score(y2_test, y2_pred))
print("Recall:", sk.metrics.recall_score(y2_test, y2_pred))
print()
print(sk.metrics.classification_report(y2_test, y2_pred))
print()


# Plotting confusion matrix for raw logistic
sk.metrics.plot_confusion_matrix(logreg_raw, X2_test, y2_test, cmap='Blues')
plt.title("Raw Logistic Confusion Matrix (2018)")
plt.show()


# Model 2: Lasso Features
logreg_lass = LogisticRegression(penalty='l1', C=1/10,  solver='liblinear', max_iter=10000)
logreg_lass.fit(X2_train[selected_features1],y2_train)
y2_pred=logreg_lass.predict(X2_test[selected_features1])


# Printing scores and classification report
print()
print("Lasso-Selected Logistic Regression Scores:")
print("Accuracy:", sk.metrics.accuracy_score(y2_test, y2_pred))
print("Precision:", sk.metrics.precision_score(y2_test, y2_pred))
print("Recall:", sk.metrics.recall_score(y2_test, y2_pred))
print()
print(sk.metrics.classification_report(y2_test, y2_pred))
print()


# Plotting confusion matrix for lasso logistic
sk.metrics.plot_confusion_matrix(logreg_lass, X2_test[selected_features1], y2_test, cmap='Blues')
plt.title("Lasso Logistic Confusion Matrix (2018)")
plt.show()


# Model 3: RFE Features
logreg_rfe = LogisticRegression(penalty='l1', C=1/10,  solver='liblinear', max_iter=10000)
logreg_rfe.fit(X2_train[selected_features2],y2_train)
y2_pred=logreg_rfe.predict(X2_test[selected_features2])


# Printing scores and classification report
print()
print("RFE-Selected Logistic Regression Scores:")
print("Accuracy:", sk.metrics.accuracy_score(y2_test, y2_pred))
print("Precision:", sk.metrics.precision_score(y2_test, y2_pred))
print("Recall:", sk.metrics.recall_score(y2_test, y2_pred))
print()
print(sk.metrics.classification_report(y2_test, y2_pred))
print()


# Plotting confusion matrix
sk.metrics.plot_confusion_matrix(logreg_rfe, X2_test[selected_features2], y2_test, cmap='Blues')
plt.title("RFE Logistic Confusion Matrix (2018)")
plt.show()


# Plotting comparisons of the ROC curves between linear and logistic regression models
raw_plot = sk.metrics.plot_roc_curve(logreg_raw, X2_test, y2_test)
lass_plot = sk.metrics.plot_roc_curve(logreg_lass, X2_test[selected_features1], y2_test, ax=raw_plot.ax_)
rfe_plot = sk.metrics.plot_roc_curve(logreg_rfe, X2_test[selected_features2], y2_test, ax=raw_plot.ax_)
labels = ['Raw Logistic', 'Lasso Logistic', 'RFE Logistic']
plt.legend(labels=labels)
plt.title("ROC Curves with Various Feature Selection Methods (2018)")
plt.show()










