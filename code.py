import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

#load all datasets
df_loan = pd.read_csv('loan.csv')
df_payment = pd.read_csv('payment.csv')
df_underwriting = pd.read_csv('clarity_underwriting_variables.csv',low_memory=False) #data file contains mixed types

#join df_loan and df_payment by loadId
df1 = pd.merge(df_loan,df_payment, how = 'inner', on = 'loanId')

#underwritingid in clarity_underwriting_variables.csv is equivalent to clarityFraudId in loan.csv
df_underwriting.rename(columns = {'underwritingid':'clarityFraudId'}, inplace = True) #rename underwritingid to clarityFraudId

#filter out applicants who don't have a clarityFraudId by merging df_underwriting and df1
df = pd.merge(df1,df_underwriting, how = 'inner', on = 'clarityFraudId')

#check for missing values
missing_values_sum = df.isnull().sum()
#print(missing_values_sum) 
pd.set_option('display.max_rows', 80)

#Aim: To see the distribution of Fraud Scores
#Drop the rows with missing values only for clearfraudscore column (in this case: 1599 rows)
#mod_df = modified dataframe
mod_df = df.dropna( how='any',
                    subset=['clearfraudscore'])

#plot a histogram that shows a proportion of each range of scores
fraudscores = mod_df.clearfraudscore
plt.hist(fraudscores, bins = 10,weights=np.ones(len(fraudscores)) / len(fraudscores), edgecolor='black', linewidth=1.2)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) #change frequency of occurrence to percentage
plt.xlabel('Fraud Scores')
plt.ylabel('Percentage of loan applicants')
plt.title('Distribution of Fraud Scores')
plt.locator_params(axis="x", nbins=10) #increase number of ticks on axes
plt.locator_params(axis="y", nbins=10)
plt.show()

import statistics
meanscore = statistics.mean(fraudscores)
print(meanscore)
modescore = statistics.mode(fraudscores)
print(modescore)
medianscore = statistics.median(fraudscores)
print(medianscore)

#mean is lower than median implying graph is negatively skewed

#removing outliers
#produce a nice summary of the dataset using descriptive statistics
df.describe()

#Aim: To find a relationship between loan status and loan amount
#Create a second modified dataframe with only the relevant columns and plot a boxplot for each loan status
mod_df2 = df[['loanStatus','loanAmount']]
fig = mod_df2.assign(index=mod_df2.groupby('loanStatus').cumcount()).pivot('index','loanStatus','loanAmount').plot(kind='box')
fig = plt.gcf()
fig.set_size_inches(30,10.5)

#alternative method
import seaborn as sb
mod_df2 = df[['loanStatus','loanAmount']]
fig = sb.boxplot(x='loanStatus',y='loanAmount',data = mod_df2,palette ='hls')
fig = plt.gcf()
fig.set_size_inches(30,10.5)

