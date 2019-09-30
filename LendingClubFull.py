# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:34:19 2019

@author: g6falle
"""
import pandas as pd
import numpy as np
import os

path="C:/Users/g6falle/Desktop/Folders/Python/LC"

os.chdir(path)

old_data = pd.read_csv('loan_old.csv',error_bad_lines = False,low_memory = False)

dtype= old_data.dtypes

new_data = pd.read_csv('loan_new.csv',error_bad_lines = False,low_memory=False)

dtype= new_data.dtypes


# Making seprate Df for grade G
G = new_data['grade'] == 'G'
grade_g_2M = new_data[G]

G1 = old_data['grade'] == 'G'
grade_g_88T = old_data[G1]

p = list(grade_g_88T['loan_status'].values)

# Grade_G with loan status Does not meet the credit policy. Status: Charged off & Fully Paid
DN_fullpaid = grade_g_88T['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid'
g_DN_fullpaid = grade_g_88T[DN_fullpaid]

DN_chargedoff = grade_g_88T['loan_status'] == 'Does not meet the credit policy. Status:Charged Off'
g_DN_chargedoff= grade_g_88T[DN_chargedoff]


doNotMeet = grade_g_88T['loan_status'].str.startswith('Does not')
grade_g_DonotMeet = grade_g_88T[doNotMeet]
# Making seprate Df for description
old_data['desc'].isnull().sum()/len(old_data)*100
old_data_desc = old_data.dropna(axis = 0, subset = ['desc'])

#slicing the dataset 
a = new_data.loc[:200000]
frames = [old_data,a]

# Merging the two files
df = pd.concat(frames, sort = False)

df = df
#OG SS
ss_og = df.describe()
#OG Skew
sk_og = df.skew()


#Removing rows for ID 
type(df)
df = df.dropna(axis=0, subset=['id'])


#Removing rows for grade = G and loan status = "Does not Meet the credit policy. Status: Fully paid & charged off"

df = df[(df['grade'] != "G") & (df['loan_status'] != 'Does not meet the credit policy. Status:Charged Off') & (df['loan_status'] != 'Does not meet the credit policy. Status:Fully Paid')]


# Percentage of null values
Null_percentage = df.isnull().sum()/len(df)*100



#selecting the columns which are less than 10% null value

df_clean= df.loc[:, df.isnull().mean()<0.1]
df_clean = df_clean[(df_clean['loan_status'] != "Issued")]

#Remove people earning more than 250,000$

df_clean = df_clean[(df_clean['annual_inc'] < 250000)&(df_clean['annual_inc'] != 0)]

df_head1 = df_clean.head()

#Revol high credit dis

df_clean = df_clean[(df_clean['total_rev_hi_lim'] < 250000)]

#Total Cur Bal under 600k only

df_clean= df_clean[df_clean['tot_cur_bal'] < 600000]

#Remove Revol balance 

df_clean= df_clean[df_clean['revol_bal'] < 85000]

#dffun['delinq_2yrs'] less than 3

df_clean = df_clean[df_clean['delinq_2yrs'] <3]

#KEEP ONLY RENT,MORTGAG,OWN

df_clean = df_clean[(df_clean['home_ownership']=='RENT')|(df_clean['home_ownership']=='MORTGAGE')|(df_clean['home_ownership'] == 'OWN')]

#Revol_Uti

df_clean = df_clean[df_clean['revol_util'] < 100]

# Open_accounts drop over 30.35

df_clean = df_clean[df_clean['open_acc'] < 30.35]


#Making grade A Late_Fee Average for loan_status = "Fully paid"

fullyPaid_A = df_clean[(df_clean['grade'] == 'A') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_A_LateFee = fullyPaid_A[(fullyPaid_A['total_rec_late_fee'] > 0)]
fullPaid_A_Mean = fullyPaid_A_LateFee['total_rec_late_fee'].mean()


#Making grade B Late_Fee Average for loan_status = "Fully paid"


fullyPaid_B = df_clean[(df_clean['grade'] == 'B') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_B_LateFee = fullyPaid_B[(fullyPaid_B['total_rec_late_fee'] > 0)]
fullPaid_B_Mean = fullyPaid_B_LateFee['total_rec_late_fee'].mean()



#Making grade C Late_Fee Average for loan_status = "Fully paid"


fullyPaid_C = df_clean[(df_clean['grade'] == 'C') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_C_LateFee = fullyPaid_C[(fullyPaid_C['total_rec_late_fee'] > 0)]
fullPaid_C_Mean = fullyPaid_C_LateFee['total_rec_late_fee'].mean()



#Making grade D Late_Fee Average for loan_status = "Fully paid"


fullyPaid_D = df_clean[(df_clean['grade'] == 'D') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_D_LateFee = fullyPaid_D[(fullyPaid_D['total_rec_late_fee'] > 0)]
fullPaid_D_Mean = fullyPaid_D_LateFee['total_rec_late_fee'].mean()

#Making grade E Late_Fee Average for loan_status = "Fully paid"

fullyPaid_E= df_clean[(df_clean['grade'] == 'E') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_E_LateFee = fullyPaid_E[(fullyPaid_E['total_rec_late_fee'] > 0)]
fullPaid_E_Mean = fullyPaid_E_LateFee['total_rec_late_fee'].mean()


#Making grade F Late_Fee Average for loan_status = "Fully paid"

fullyPaid_F = df_clean[(df_clean['grade'] == 'F') & (df_clean['loan_status'] == "Fully Paid")]
fullyPaid_F_LateFee = fullyPaid_F[(fullyPaid_F['total_rec_late_fee'] > 0)]
fullPaid_F_Mean = fullyPaid_F_LateFee['total_rec_late_fee'].mean()
# Making a seprate Dict for Grade Late_fee average for loan_status = "Fully paid"

full_Mean =  {'A': fullPaid_A_Mean, 'B': fullPaid_B_Mean, 'C': fullPaid_C_Mean, 'D': fullPaid_D_Mean , 'E': fullPaid_E_Mean , 'F': fullPaid_F_Mean}


zz = df_clean['loan_status']


#Making Grade A late_fee average for loan_status active
Active_A = df_clean[(df_clean['grade'] == 'A') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_A_LateFee = Active_A[(Active_A['total_rec_late_fee'] > 0)]
Active_A_Mean = Active_A_LateFee['total_rec_late_fee'].mean()

#Making Grade B late_fee average for loan_status active
Active_B = df_clean[(df_clean['grade'] == 'B') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_B_LateFee = Active_B[(Active_B['total_rec_late_fee'] > 0)]
Active_B_Mean = Active_B_LateFee['total_rec_late_fee'].mean()


#Making Grade C late_fee average for loan_status active
Active_C = df_clean[(df_clean['grade'] == 'C') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_C_LateFee = Active_C[(Active_C['total_rec_late_fee'] > 0)]
Active_C_Mean = Active_C_LateFee['total_rec_late_fee'].mean()

#Making Grade D late_fee average for loan_status active
Active_D = df_clean[(df_clean['grade'] == 'D') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_D_LateFee = Active_D[(Active_D['total_rec_late_fee'] > 0)]
Active_D_Mean = Active_D_LateFee['total_rec_late_fee'].mean()

#Making Grade E late_fee average for loan_status active
Active_E = df_clean[(df_clean['grade'] == 'E') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_E_LateFee = Active_E[(Active_E['total_rec_late_fee'] > 0)]
Active_E_Mean = Active_E_LateFee['total_rec_late_fee'].mean()

#Making Grade E late_fee average for loan_status active
Active_F = df_clean[(df_clean['grade'] == 'F') & (df_clean['loan_status'] != "Fully Paid") & (df_clean['loan_status'] !="Charged Off") & (df_clean['loan_status'] != "Default")]
Active_F_LateFee = Active_F[(Active_F['total_rec_late_fee'] > 0)]
Active_F_Mean = Active_F_LateFee['total_rec_late_fee'].mean()


#Function for making new column Active_loan_average_late_fee
def change_active (c):
    if (c['grade'] == 'A') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_A_Mean
    elif (c['grade'] == 'B') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_B_Mean
    elif (c['grade'] == 'C') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_C_Mean
    elif (c['grade'] == 'D') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_D_Mean
    elif (c['grade'] == 'E') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_E_Mean
    elif (c['grade'] == 'F') & (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return Active_F_Mean
    else:
        return 0    

# Applying the function
df_clean['Active_loan_Average_late_fee']=df_clean.apply(change_active, axis = 1)
df_clean['Active_loan_Average_late_fee'].astype(float)

#Function for making new column Fully_Paid_Loan_Average_late_fee
def change_fullypaid (x):
    if (x['grade'] == 'A') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_A_Mean
    elif (x['grade'] == 'B') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_B_Mean
    elif (x['grade'] == 'C') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_C_Mean
    elif (x['grade'] == 'D') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_D_Mean
    elif (x['grade'] == 'E') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_E_Mean
    elif (x['grade'] == 'F') & (x['loan_status'] == 'Fully Paid'):
        return fullPaid_F_Mean
    else:
        return 0 

# Applying the function
df_clean['FullyPaid_loan_Average_late_fee'] = df_clean.apply(change_fullypaid, axis =1)
df_clean['FullyPaid_loan_Average_late_fee'].astype(float)


#Creating late Pyament modifier

grade_A_pm = df_clean[(df_clean['grade']=='A')]
lpm_A = (sum(grade_A_pm['total_rec_late_fee'])/ sum(grade_A_pm['funded_amnt']))*100

grade_B_pm = df_clean[(df_clean['grade']=='B')]
lpm_B = (sum(grade_B_pm['total_rec_late_fee'])/ sum(grade_B_pm['funded_amnt']))*100

grade_C_pm = df_clean[(df_clean['grade']=='C')]
lpm_C = (sum(grade_C_pm['total_rec_late_fee'])/ sum(grade_C_pm['funded_amnt']))*100

grade_D_pm = df_clean[(df_clean['grade']=='D')]
lpm_D = (sum(grade_D_pm['total_rec_late_fee'])/ sum(grade_D_pm['funded_amnt']))*100

grade_E_pm = df_clean[(df_clean['grade']=='E')]
lpm_E = (sum(grade_E_pm['total_rec_late_fee'])/ sum(grade_E_pm['funded_amnt']))*100

grade_F_pm = df_clean[(df_clean['grade']=='F')]
lpm_F = (sum(grade_F_pm['total_rec_late_fee'])/ sum(grade_F_pm['funded_amnt']))*100


#Function for making new column Late Payment Modifier
def paymentModifier (i):
    if i['grade'] == 'A':
        return lpm_A
    elif i['grade'] == 'B':
        return lpm_B
    elif i['grade'] == 'C':
        return lpm_C
    elif i['grade'] == 'D':
        return lpm_D
    elif i['grade'] == 'E':
        return lpm_E
    elif i['grade'] == 'F':
        return lpm_F


#Applying the function
df_clean['Late_Payment_Modifier'] = df_clean.apply(paymentModifier,axis = 1)
df_clean['Late_Payment_Modifier'].astype(float)


dffun = pd.DataFrame(df_clean)

# Creating issue_date column from issue
dffun["issue_date"] = pd.to_datetime(dffun["issue_d"]).dt.date

df_head = df_clean.head()

dffun['term'].astype(str)


dffun["num_month"] = dffun.term.str.extract('(\d+)').astype(str)
dffun['num_month'] = dffun.num_month.astype(int)

# Funing came from Lending Club Yes or No, not from investors

dffun['lc_fun_amt'] = dffun["funded_amnt"] - dffun["funded_amnt_inv"]

def f(row):
    if row['lc_fun_amt'] > 0:
        val = 1
    else:
        val = 0
    return val


dffun["lc_fun"] = dffun.apply(f,axis=1)

dfhead = dffun.head(100)

# Total amount paid for active loan
def total (c):
    if (c['loan_status'] != "Fully Paid" ) & (c['loan_status'] != 'Charged Off') & (c['loan_status'] != "Default"):
        return c['installment'] * c['num_month']
    else:
        return 0


dffun["total_payment_asu"] = dffun.apply(total, axis =1)

#late_pay for active loan
def late (c):
    if c['total_payment_asu'] != 0:
        return c['funded_amnt'] * c['Late_Payment_Modifier']
    else:
        return 0

dffun['late_pay'] = dffun.apply(late, axis = 1)

#total_final _pay 
def total_final (c):
    if c['late_pay'] != 0:
        return c['total_payment_asu'] + c['late_pay']
    else: 
        return 0
    

dffun['total_pay_f'] = dffun.apply(total_final, axis = 1)

# return_per

def return_per(c):
    if c['total_pay_f'] != 0:
        return c['total_pay_f'] / c['funded_amnt']
    else:
        return c['total_pymnt'] / c['funded_amnt']

dffun['return_per'] = dffun.apply(return_per, axis = 1)

dfhead = dffun.head(100)


# The loan made at least a 15% profit for the investor

def f(row):
    if row['return_per'] > 1.15:
        val = 1
    else:
        val = 0
    return val


dffun["over_15"] = dffun.apply(f,axis=1)
count = dffun.over_15.value_counts()
len(dffun)

########################################### Average intrest rate by Grade ####################################

a = dffun.groupby('grade')['int_rate'].mean()

########################################### Average intrest rate by Grade ####################################


#writing to csv

dffun.to_csv("cleaned__file_updated.csv", index = False)


# ppl who have charged off or default replace total_pymnt = total_pymnt_f

dffun.loc[(dffun['loan_status']=='Charged Off') | (dffun['loan_status']== 'Default')|(dffun['loan_status']=='Fully Paid'),'total_pay_f'] = dffun['total_pymnt']

dffun.describe()

#T-Test 

homevsloan = dffun[(dffun['home_ownership'] == 'OWN')] 
own= homevsloan['loan_amnt']
homevsloan1 = dffun[(dffun['home_ownership'] == 'MORTGAGE')]
mortage =homevsloan1['loan_amnt']
homevsloan2 = dffun[(dffun['home_ownership'] == 'RENT')]

own = homevsloan.describe()
mortage = homevsloan1.describe()
rent = homevsloan2.describe()
len(homevsloan1)



# h0 = own> = mortage h1 = mortage > own

from scipy import stats


print(stats.ttest_ind(own,mortage, equal_var = False)) # p < .001 reject h0 , ppl who have mortage tend to get more loan amount

#case 2 


debt_amount = dffun[(dffun['purpose'] == 'debt_consolidation')]
debt = debt_amount['loan_amnt']
house_amount = dffun[(dffun['purpose'] == 'house')]
house = house_amount['loan_amnt']

#ho = house > = debt h1 = debt > house

print(stats.ttest_ind(debt, house,  equal_var = False)) # p <.001 reject h0, ppl  ask more loan amount for debt consolidation when compared to house

# case 3 

small_amount = dffun[(dffun['purpose'] == 'small_business')]
small = small_amount['loan_amnt']

#ho = small > = debt h1 = debt > small

print(stats.ttest_ind(debt, small,  equal_var = False)) # p value = 0.08563 reject h1, so people ask higher loan amount for small business

# case 4 

emp_length = dffun[(dffun['emp_length'] == '10+ years')]
ten_year = emp_length['int_rate']
seven_year_amnt = dffun[(dffun['emp_length'] == '7 years')]
seven_year = seven_year_amnt['int_rate']

#ho = ten_year > = 7_year h1 = 7_year > 10year

print(stats.ttest_ind(ten_year,seven_year,  equal_var = False)) # p value < .001 reject ho, ppl who have 7year have more interest rate

df_backup = dffun

dffun = df_backup

#Deleting columns 

counttest = dffun.pub_rec.value_counts()
del dffun['out_prncp']
del dffun['total_pymnt']
del dffun['collection_recovery_fee']
del dffun['total_pymnt_inv']
del dffun['total_rec_prncp']
del dffun['total_rec_int']
del dffun['total_rec_late_fee']
del dffun['recoveries']
del dffun['last_pymnt_amnt']
del dffun['id']
del dffun['member_id']
del dffun['tot_coll_amt']
del dffun['policy_code']
del dffun['Active_loan_Average_late_fee']
del dffun['FullyPaid_loan_Average_late_fee']
del dffun['Late_Payment_Modifier']
del dffun['late_pay']
del dffun['total_payment_asu']
del dffun['term']
del dffun['url']
del dffun['zip_code']
del dffun['collections_12_mths_ex_med']

# dropping the rows whose paymnt plan = y because only 10 ppl have yes in the whole population
dffun = dffun[dffun.pymnt_plan !='y']

#dropping the rows whose application type is "joint" before dropping the whole columns

dffun = dffun[dffun.application_type != 'JOINT']


# Removing the column application type, coz our DF everyone is indiviual now
del dffun['application_type']


#converting null into zero
dffun['total_rev_hi_lim'].fillna(0, inplace = True)

dffun_ss = dffun.describe()


# Skwness 

dfsk=pd.DataFrame()

dfsk = pd.DataFrame(dffun.skew())
dfsk.columns = ['Skew']
dfsk["col"] = dfsk.index

dfsk=dfsk.reset_index()

dfsk=dfsk.drop(['index'],axis=1)

dfhigsk = dfsk.loc[dfsk['Skew'] > 3]

dflowsk = dfsk.loc[dfsk['Skew'] < -3]

dflist = dfhigsk['col'].tolist()

df_skew = dffun[dflist]

dflg = np.log1p(df_skew)

dflguse =pd.DataFrame()

dflguse = pd.DataFrame(dflg.skew())

dflguse.columns = ['Skew']

dflguse["col"] = dflguse.index

dflguse=dflguse.reset_index()

dflguse = dflguse.drop(['index'],axis=1)

dflgkeep = dflguse[(dflguse['Skew'] < 3) & (dflguse['Skew'] > -3)]

df_log_cols = dflgkeep['col'].tolist()

df_log_list = dflg[df_log_cols]

dflogvar = df_log_list.add_prefix('log_')

dflgkeep['col'] = 'log_' + dflgkeep['col']

dffun_ = pd.concat([dffun,dflogvar],axis=1, sort = False)

df_last = dffun_.head()
frm = [dflgkeep,dflowsk]

df_skew_final = pd.concat(frm, sort = False)

df_head_skew = dffun_.head()

dffun_.to_csv("aa.csv", index = False)



###################################################################After Log###########################
vissk = dffun_[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 'installment','annual_inc', 'dti','inq_last_6mths','open_acc','pub_rec','revol_bal','revol_util','total_acc','tot_cur_bal','total_rev_hi_lim','return_per','total_pay_f','log_pub_rec']]
split = 18

data=vissk.iloc[:,:split]

cols=data.columns

import seaborn as sns
import matplotlib.pyplot as plt

n_cols = 2
n_rows = 6

for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1,ncols=n_cols, figsize=(12,8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=data,ax=ax[j])
        
#################################################################Pior before log########################


data=dffun_[['pub_rec']]

size=1

cols=data.columns

n_cols = 2
n_rows = 6

for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1,ncols=n_cols, figsize=(12,8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=data,ax=ax[j])


##############################################log

data=dffun_[['log_pub_rec','pub_rec']]

size=1

cols=data.columns

n_cols = 2
n_rows = 6

for i in range(n_rows):
    fg, ax = plt.subplots(nrows=1,ncols=n_cols, figsize=(12,8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=data,ax=ax[j])
        
################################################
size = 15
data = vissk

data_corr = data.corr()

cols=data.columns


# Set the threshold to select only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []


#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index
            
            #Sort to show higher ones first            
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
    
# PairPlot

for v,i,j in s_corr_list:
    sns.pairplot(data, size=6, x_vars=cols[i],y_vars=cols[j])
plt.show()

#Visulaize Corrlations 

corr = data.corr()
import seaborn as sns

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation= 'vertical',
    horizontalalignment='right'
);



###################################################



data=dffun_[['grade','sub_grade','emp_title',
                 'emp_length','home_ownership','verification_status','issue_d','loan_status', 'pymnt_plan', 'purpose', 'title',
                 'addr_state','earliest_cr_line','last_credit_pull_d','issue_date',  
                 'lc_fun',
               'over_15']]

# If Emp Length is null then use 10+ years because of mode

dffun_['emp_length'] = dffun_['emp_length'].replace(np.nan, '10+ Years', regex=True)
count_nan = len(df) - df.count()

# Selecting number of years emp

dffun_["yr_emp_int"] = dffun_.emp_length.str.extract('(\d+)').astype(int)

#Create variable if emp is over 5 years then 1 else 0

def f(row):
    if row['yr_emp_int'] > 5:
        val = 1
    else:
        val = 0
    return val

dffun_["emp_more_fiv"] = dffun_.apply(f,axis=1)

################################################# ANOVA - SAMPLE ONLY FIRST 100 ROWS FOR DEMO ###########################################################

dfsmp = dffun_.head(100)

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import seaborn as sns
aa = dfsmp[['grade','return_per']]
aa.columns = ['grade', 'value']


sns.countplot(x="value", hue="grade", data=aa)



sns.boxplot(x ="grade" ,y ="value" ,  data = aa, palette = "Set3")
model = ols('value ~ C(grade)', data = aa).fit()
anova_table = sm.stats.anova_lm(model, typ =1 )
anova_table

m_comp = pairwise_tukeyhsd(endog = aa['value'], groups = aa['grade'] , alpha =0.05)
print(m_comp)

################################################# ANOVA - SAMPLE ONLY FIRST 100 ROWS FOR DEMO ###########################################################


################################################# CREATE A FINAL DF ###########################################################

df_final = dffun_[['loan_amnt','dti','int_rate','annual_inc','inq_last_6mths','open_acc','log_pub_rec','revol_bal',
                   'revol_util','tot_cur_bal','lc_fun_amt','delinq_2yrs','total_pay_f','acc_now_delinq','out_prncp_inv','num_month',
                   'emp_more_fiv','lc_fun','grade','sub_grade', 'home_ownership','verification_status','purpose','addr_state','return_per','installment','over_15'
                   ]]

################################################# CREATE A FINAL DF ###########################################################

#State
states = {
        'AK': 'O',
        'AL': 'S',
        'AR': 'S',
        'AS': 'O',
        'AZ': 'W',
        'CA': 'W',
        'CO': 'W',
        'CT': 'N',
        'DC': 'N',
        'DE': 'N',
        'FL': 'S',
        'GA': 'S',
        'GU': 'O',
        'HI': 'O',
        'IA': 'M',
        'ID': 'W',
        'IL': 'M',
        'IN': 'M',
        'KS': 'M',
        'KY': 'S',
        'LA': 'S',
        'MA': 'N',
        'MD': 'N',
        'ME': 'N',
        'MI': 'W',
        'MN': 'M',
        'MO': 'M',
        'MP': 'O',
        'MS': 'S',
        'MT': 'W',
        'NA': 'O',
        'NC': 'S',
        'ND': 'M',
        'NE': 'W',
        'NH': 'N',
        'NJ': 'N',
        'NM': 'W',
        'NV': 'W',
        'NY': 'N',
        'OH': 'M',
        'OK': 'S',
        'OR': 'W',
        'PA': 'N',
        'PR': 'O',
        'RI': 'N',
        'SC': 'S',
        'SD': 'M',
        'TN': 'S',
        'TX': 'S',
        'UT': 'W',
        'VA': 'S',
        'VI': 'O',
        'VT': 'N',
        'WA': 'W',
        'WI': 'M',
        'WV': 'S',
        'WY': 'W'
}


df_final['addr_state']= df_final['addr_state'].map(states)
count = df_final.addr_state.value_counts()

#Month


def mth(xa):
    if xa['num_month'] == 36:
        val = 1
    else:
        val=  0
    return val

df_final['num_month'] = df_final.apply(mth,axis = 1)

count = df_final.emp_more_fiv.value_counts()

df_final.to_csv('asofnow.csv',index = False)

# continous variable

df_final_cont = dffun_[['loan_amnt','int_rate', 'annual_inc', 'inq_last_6mths','open_acc','log_pub_rec','revol_bal',
                   'revol_util','tot_cur_bal','lc_fun_amt','delinq_2yrs','total_pay_f','acc_now_delinq','out_prncp_inv']]


# Boolean 

df_final_boo = dffun_[[
        'emp_more_fiv',
        'lc_fun',]]


#Categorical_varaible 

df_final_cat = df_final[['grade','sub_grade','home_ownership','verification_status','purpose','addr_state']]

#Create Variable for Payment to Income Ratio

df_final['per_month'] = df_final['annual_inc']/12
df_final['pay_to_income'] = df_final['installment']/df_final['per_month']


# X Variable

#x = df_final[['loan_amnt','dti','annual_inc', 'inq_last_6mths','open_acc','log_pub_rec','revol_util','delinq_2yrs','acc_now_delinq','pay_to_income','emp_more_fiv','lc_fun' , 'grade']]


# Y VARIABLE

#Y = df_final[['over_15']]

x=pd.read_csv(


# DUMMY VARIABLE IF GRADE IS ADDED INTO X 

xc = pd.get_dummies(x, columns = ['grade'])


# Splitting the dataset

from sklearn.model_selection import train_test_split
x_train, x_test ,Y_train ,Y_test = train_test_split(xc,Y, test_size= 0.30, random_state = 0)

##################################################################fitting the logisitic regression to the training set################################
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(fit_intercept = True, random_state = 0)
classifier.fit(x_train,Y_train)

#prediciting the test set results
Y_pred = classifier.predict(x_test)


#making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

print('Accuracy:{:.2f}'.format(classifier.score(x_test,Y_test)))


#f-1 score
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


#ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test, classifier.predict(x_test))
fpr, tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.savefig('Log_ROC')
plt.show()

##################################################################fitting the logisitic regression to the training set################################


#################################################### RANDOM FOREST OVERFITTING ################################################

# Fitting Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier_r = RandomForestClassifier(n_estimators = 100 ,  random_state =0, n_jobs = -1, oob_score = True, bootstrap = True )
classifier_r.fit(x_train,Y_train)

#SCORE

print('R^2 Training Score: {:.4f} \nOOB Score: {:.4f} \nR^2 Test Score: {:.4f}'.format(classifier_r.score(x_train, Y_train), 
                                                                                             classifier_r.oob_score_,
                                                                                             classifier_r.score(x_test, Y_test)))




################################################## RANDOM FOREST NO-OVERFITTING #############################################

# Fitting Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier_r1 = RandomForestClassifier(n_estimators = 300 , max_depth = 10,  random_state =0, n_jobs = -1, oob_score = True, bootstrap = True )
classifier_r1.fit(x_train,Y_train)


#SCORE
print('R^2 Training Score: {:.4f} \nOOB Score: {:.4f} \nR^2 Test Score: {:.4f}'.format(classifier_r1.score(x_train, Y_train), 
                                                                                             classifier_r1.oob_score_,
                                                                                             classifier_r1.score(x_test, Y_test)))



#SCORE FOR CROSS VALIDATION
from sklearn.model_selection import cross_val_score
score_RF = cross_val_score(classifier_r1, xc, Y, cv =5, scoring = 'accuracy') # CHANGE X variable accordingly
print("Accuracy: %0.2f (+/- %0.2f)" % (score_RF.mean(), score_RF.std() *2))
df_final_list = list(xc.columns)

# CONFUSION MATRIX PLOT

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
y_pred_cl = classifier_r1.predict(x_test)
mat = confusion_matrix(Y_test, y_pred_cl)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


######################################################## TREEE PLOT USE WITH MAX_DEPTH =3 for Better Picture #############

col_list = list(xc.columns) ############ CHANGE X VARIABLE ACCORDINGLY

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = classifier_r1.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = classifier_r1.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = col_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


################################################################ IMPORTANCE OF VARIABLE #################################

# Get numerical feature importances
importances = list(classifier_r1.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(x, round(importance, 2)) for x, importance in zip(col_list, importances)]  ### CHANGE X VARIABLE ACCORDIGNLY
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



################################################################# PLOTING VARIABLE IMPORTANCE #######################

# Import matplotlib for plotting 
import matplotlib.pyplot as plt
%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, df_final_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


################################################################ PREDICTING FOR 2 ROWS ###################################


from treeinterpreter import treeinterpreter as ti

selected_rows = [31, 85]
selected_df = x_train.iloc[selected_rows,:].values
prediction, bias, contributions = ti.predict(regressor_r, selected_df)

aa = np.array(Y_train)

for i in range(len(selected_rows)):
    print("Row", selected_rows[i])
    print("Prediction:", prediction[i][0])                   , print( 'Actual Value:', aa[selected_rows[i]])
    print("Bias (trainset mean)", bias[i])
    print("Feature contributions:")
    for c, feature in sorted(zip(contributions[i], 
                                 x_train.columns), 
                             key=lambda x: np.logical_not(x[0]).any()):
        print(feature, round(c[i],2))
    print("-"*20)

