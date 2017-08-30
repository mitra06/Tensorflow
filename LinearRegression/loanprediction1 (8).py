
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
get_ipython().magic('matplotlib inline')


# In[5]:

train=pd.read_csv("F:\DataScienceAnaconda\Hackathon\loantrain.csv")
train.head(10)


# In[101]:

train.dtypes


# In[109]:

colTypes = pd.read_csv('F:\DataScienceAnaconda\Hackathon\datatypes.csv')
colTypes


# In[110]:

for i, row in colTypes.iterrows():
    if row['type']=="categorical":
        train[row['feature']]=train[row['feature']].astype(np.object)
    elif row['type']=="continuous":
        train[row['feature']]=train[row['feature']].astype(np.float)
train.dtypes


# In[6]:

train.tail(10)


# In[8]:

train.describe()


# In[10]:

train['ApplicantIncome'].mean()


# In[11]:

train['Gender'].unique()


# In[12]:

train['Education'].unique()


# In[14]:

train['Self_Employed'].unique()


# In[16]:

train['Credit_History'].unique()


# In[17]:

train['Property_Area'].unique()


# In[18]:

train['Loan_Status'].unique()


# In[21]:

# Distribution Analysis

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(train['ApplicantIncome'], bins = 50, range = (train['ApplicantIncome'].min(),train['ApplicantIncome'].max()))
plt.title('Income distribution')
plt.xlabel('ApplicantIncome')
plt.ylabel('Count of Person')
plt.show()


# In[23]:

train.boxplot('ApplicantIncome')
plt.show()


# In[30]:

train['ApplicantIncome'].max()


# In[31]:

train['ApplicantIncome'].min()


# In[38]:

train.groupby(pd.cut(train['ApplicantIncome'], np.arange(1000,10000,500))).count()


# In[43]:

train.boxplot(column='ApplicantIncome', by = 'Gender')
plt.show()


# In[44]:

train.boxplot(column='ApplicantIncome', by = 'LoanAmount')
plt.show()


# In[51]:

temp3 = pd.crosstab([train.Gender, train.Education], train.Married.astype(bool))
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.show()


# In[52]:

temp3 = pd.crosstab([train.Gender, train.Education], train.Self_Employed.astype(bool))

temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.show()


# In[62]:

gg=train.groupby('Gender')['ApplicantIncome'].sum()
gg


# In[63]:

gg=train.groupby('Gender')['ApplicantIncome'].sum()
gg


# In[64]:

train["Gender"].value_counts()


# In[66]:

train["Self_Employed"].value_counts()


# In[68]:

gg=train.groupby('Gender')['Self_Employed'].value_counts()
gg


# In[71]:

temp3 = pd.crosstab([train.Gender, train.Self_Employed], train.Loan_Status.astype(bool))
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.show()


# In[74]:

train.loc[(train["Gender"]=="Female") & (train["Education"]=="Not Graduate") & (train["Loan_Status"]=="Y"), ["Gender","Education","Loan_Status"]]


# In[75]:

def num_missing(x):
    return sum(x.isnull())
train.apply(num_missing, axis=0)


# In[76]:

def num_missing(x):
    return sum(x.isnull())
train.apply(num_missing, axis=1)


# In[78]:

def num_missing(x):
    return sum(x.isnull())
train.apply(num_missing, axis=1).head()


# In[86]:

impute_grps = train.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
impute_grps


# In[87]:

pd.crosstab(train["Credit_History"],train["Loan_Status"],margins=True)


# In[94]:

train.boxplot(column="ApplicantIncome",by="Dependents")


# In[96]:

train.hist(column="ApplicantIncome",by="Loan_Status",bins=30)


# In[97]:

def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

#Binning age:
cut_points = [90,140,190]
labels = ["low","medium","high","very high"]
train["LoanAmount_Bin"] = binning(train["LoanAmount"], cut_points, labels)
pd.value_counts(train["LoanAmount_Bin"], sort=False)


# In[112]:

train.apply(lambda x: sum(x.isnull()),axis=0) 


# In[117]:

train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)


# In[114]:

train.apply(lambda x: sum(x.isnull()),axis=0) 


# In[118]:

train['Self_Employed'].value_counts()


# In[120]:

train['Self_Employed'].fillna('No',inplace=True)


# In[121]:

train['Self_Employed'].value_counts()


# In[123]:

train['Education'].value_counts()


# In[126]:

train['Dependents'].value_counts()


# In[125]:

train['Credit_History'].value_counts()


# In[124]:

train['Married'].value_counts()


# In[130]:

train['Education'].fillna('Graduate',inplace=True)


# In[131]:

train['Education'].value_counts()


# In[132]:

train['Married'].fillna('Yes',inplace=True)


# In[133]:

train['Married'].value_counts()


# In[134]:

train['Dependents'].fillna('0',inplace=True)


# In[135]:

train['Dependents'].value_counts()


# In[139]:

train['Credit_History'].fillna('1.000000',inplace=True)


# In[140]:

train['Credit_History'].value_counts()


# In[141]:

train['Credit_History'].fillna('1.000000',inplace=True)


# In[142]:

train['Credit_History'].value_counts()


# In[144]:

train['Gender'].value_counts()


# In[146]:

train['Gender'].fillna('Male',inplace=True)


# In[147]:

train['Gender'].value_counts()


# In[148]:

train.apply(lambda x: sum(x.isnull()),axis=0) 


# In[150]:

table = train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
 train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[151]:

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)


# In[153]:

train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train['TotalIncome_log'] = np.log(train['TotalIncome'])
train['LoanAmount_log'].hist(bins=20) 


# In[154]:

print(train['Gender'].mode())


# In[155]:

print(train['Self_Employed'].mode())


# In[158]:

color_wheel = {'Y': "#0033ff", 
               'N': "#cf0202"}
colors = train['Loan_Status'].map(lambda x: color_wheel.get(x))
plt.scatter(train['LoanAmount'],train['Loan_Amount_Term'],c=colors)
plt.show()


# In[159]:

pd.crosstab(index=train['Gender'],
            columns=[train['Property_Area']], 
            margins=True)


# In[161]:

sg=train.groupby(['Self_Employed'])
print(sg.describe())


# In[162]:

sg=train.groupby(['Gender'])
print(sg.describe())


# In[164]:

sg=train.groupby(['Loan_Status'])
print(sg.describe())


# In[165]:

sg=train.groupby(['Dependents'])
print(sg.describe())


# In[170]:

train['Loan_Amount_Term'].hist(bins=50)
plt.show()


# In[169]:

train['LoanAmount'].hist(bins=50)
plt.show()


# In[168]:

train['CoapplicantIncome'].hist(bins=50)
plt.show()


# In[167]:

train['ApplicantIncome'].hist(bins=50)
plt.show()


# In[172]:

fig=plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(train['ApplicantIncome'],bins = 5)
plt.title('Applicant Income')
plt.xlabel('Salary Range')
plt.ylabel('No of Applicant')
plt.show()


# In[174]:

print(pd.crosstab(train['Gender'], train['Loan_Status'], margins = True ), "\n")
print(pd.crosstab(train['Married'], train['Loan_Status'], margins = True ), "\n")
print(pd.crosstab(train['Dependents'], train['Loan_Status'], margins = True ), "\n")
print(pd.crosstab(train['Education'],train['Loan_Status'], margins = True ), "\n")
print(pd.crosstab(train['Self_Employed'], train['Loan_Status'], margins = True ), "\n")
print(pd.crosstab(train['Property_Area'], train['Loan_Status'], margins = True ), "\n")


# In[10]:

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
train['Total_Income']


# In[182]:

train['Total_Income'].hist(bins=50)
plt.show()


# In[188]:

train['Total_Income'].value_counts().plot(kind='hist')
plt.show()


# In[191]:

train.hist(column="Total_Income",by= "Loan_Status",figsize= (8,8))


# In[ ]:

train.groupby(by="Total_Income").plot.hist()
plt.show()


# In[6]:

temp3 = pd.crosstab([train['Credit_History'],train['Property_Area'],train["Gender"]],train['Loan_Status'])
temp3.plot(kind="bar",stacked=True)
plt.show()


# In[7]:

temp3 = pd.crosstab([train['Property_Area'],train["Gender"]],train['Loan_Status'])
print (temp3)
temp3.plot(kind="bar",stacked=True)
plt.show()


# In[11]:

temp3 = pd.crosstab([train['Property_Area'],train["Total_Income"]],train['Loan_Status'])
print (temp3)
temp3.plot(kind="bar",stacked=True)
plt.show()


# In[13]:

train["LoanAmount"]=train["LoanAmount"].replace(np.nan,np.mean(train["LoanAmount"]),regex=True)
train["LoanAmount"]


# In[14]:

train["Loan_Amount_Term"]=train["Loan_Amount_Term"].replace(np.nan,np.mean(train["Loan_Amount_Term"]),regex=True)
train["Loan_Amount_Term"]


# In[15]:

train["Credit_History"]=train["Credit_History"].fillna(train["Credit_History"].dropna().values[0])
train["Credit_History"]


# In[17]:

train["Dependents"]=np.where(train["Married"]=="Yes",
                          train["Dependents"].replace(np.nan,2, regex=True),
                          train["Dependents"].replace(np.nan,0, regex=True))
train["Dependents"]


# In[18]:

print(train["Dependents"].values,train["Married"].values)


# In[20]:

train["Self_Employed"]=train["Self_Employed"].replace(np.nan,"No", regex=True)
train["Self_Employed"]


# In[21]:

print(train.describe(include=[object]))


# In[23]:

train["Gender"]=train["Gender"].replace(np.nan,"Male", regex=True)
train['Gender']


# In[27]:

train.groupby('Gender').hist()
plt.show()


# In[29]:

train.groupby('Credit_History').hist()
plt.show()


# In[30]:

train.groupby('Dependents').hist()
plt.show()


# In[33]:

train.groupby('Education').hist()
plt.show()


# In[34]:

train.groupby('Married').hist()
plt.show()


# In[35]:

train.groupby('Property_Area').hist()
plt.show()


# In[37]:

temp =pd.crosstab(index=[train['Credit_History']==0.0,train['Loan_Status']=='Y'],
            columns=[train['ApplicantIncome'],train['CoapplicantIncome']],
            margins=True).apply(lambda r:(r.sum()),axis=1)
temp


# In[38]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[39]:

temp =pd.crosstab(index=[train['Credit_History']==1.0,train['Loan_Status']=='Y'],
            columns=[train['ApplicantIncome'],train['CoapplicantIncome']],
            margins=True).apply(lambda r:(r.sum()),axis=1)
temp


# In[40]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[42]:

temp =pd.crosstab(index=[train['Married']=='yes',train['Loan_Status']=='N'],
            columns=[train['ApplicantIncome'],train['CoapplicantIncome']],
            margins=True).apply(lambda r:(r.sum()),axis=1)
temp


# In[43]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[44]:

temp =pd.crosstab(index=[train['Married']=='yes',train['Loan_Status']=='Y'],
            columns=[train['ApplicantIncome'],train['CoapplicantIncome']],
            margins=True).apply(lambda r:(r.sum()),axis=1)
temp


# In[45]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[46]:

temp =pd.crosstab(index=[train['Married']=='NO',train['Loan_Status']=='N'],
            columns=[train['ApplicantIncome'],train['CoapplicantIncome']],
            margins=True).apply(lambda r:(r.sum()),axis=1)
temp


# In[48]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[47]:

temp =pd.crosstab(index=[train['Married']=='yes',train['Loan_Status']=='Y'],
            columns=[train['ApplicantIncome'],train['CoapplicantIncome']],
            margins=True).apply(lambda r:(r.sum()),axis=1)
temp


# In[49]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[50]:

exception = train.loc[(train['Credit_History']==0.0) & (train['Loan_Status'] == 'Y')]
exception


# In[51]:

exception = train.loc[(train['Married']=='No') & (train['CoapplicantIncome']>0) ]
exception


# In[52]:

exception = train.loc[(train['Self_Employed']=='No') & (train['Loan_Status'] == 'Y') ]
exception


# In[53]:

temp=pd.crosstab(index=train['Gender'],
            columns=[train['Property_Area']], 
            margins=True)
temp


# In[54]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[55]:

temp=pd.crosstab(index=train['Gender'],
            columns=[train['Education']], 
            margins=True)
temp


# In[60]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[78]:

temp=pd.crosstab(index=train['Gender'],
            columns=[train['Loan_Status']], 
            margins=True)
temp


# In[61]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[58]:

temp=pd.crosstab(index=train['Gender'],
            columns=[train['Self_Employed']], 
            margins=True)
temp


# In[62]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[59]:

temp=pd.crosstab(index=train['Gender'],
            columns=[train['Credit_History']], 
            margins=True)
temp


# In[63]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[64]:

temp=pd.crosstab(index=train['Married'],
            columns=[train['Loan_Status']], 
            margins=True)
temp


# In[72]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[65]:

temp=pd.crosstab(index=train['Married'],
            columns=[train['Gender']], 
            margins=True)
temp


# In[73]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[66]:

temp=pd.crosstab(index=train['Married'],
            columns=[train['Credit_History']], 
            margins=True)
temp


# In[74]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[67]:

temp=pd.crosstab(index=train['Married'],
            columns=[train['Property_Area']], 
            margins=True)
temp


# In[75]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[70]:

temp=pd.crosstab(index=train['Married'],
            columns=[train['Education']], 
            margins=True)
temp


# In[76]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[71]:

temp=pd.crosstab(index=train['Married'],
            columns=[train['Property_Area']], 
            margins=True)
temp


# In[77]:

temp.plot(kind="bar",stacked=True)
plt.show()


# In[ ]:



