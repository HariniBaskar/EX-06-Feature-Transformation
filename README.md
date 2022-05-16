# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# Methods for Data Transformation
1. FUNCTION TRANSFORMATION
2. POWER TRANSFORMATION
3. QUANTILE TRANSFORMATION

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file

# CODE (Data_To_Transform.csv)
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  
df=pd.read_csv("Data_To_Transform.csv")  
df  
df.skew()  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Highly Positive Skew"])  
#Reciprocal Transformation  
np.reciprocal(df["Moderate Positive Skew"])  
#Square Root Transformation  
np.sqrt(df["Highly Positive Skew"])  
#Square Transformation  
np.square(df["Highly Negative Skew"])  

#POWER TRANSFORMATION:  
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])  
df  
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])  
df  
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])  
df  
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])  
df  

#QUANTILE TRANSFORMATION:  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal')  
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])  
sm.qqplot(df['Moderate Negative Skew'],line='45')  
plt.show()
sm.qqplot(df['Moderate Negative Skew_1'],line='45')  
plt.show()  
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])  
sm.qqplot(df['Highly Negative Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()  
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])  
sm.qqplot(df['Moderate Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Moderate Positive Skew_1'],line='45')  
plt.show() 
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])  
sm.qqplot(df['Highly Positive Skew'],line='45')  
plt.show()  
sm.qqplot(df['Highly Positive Skew_1'],line='45')  
plt.show()  

df.skew()  
df 
```
# OUPUT
![dtt1](https://user-images.githubusercontent.com/93427253/168624110-ef41e318-928f-42c0-bd41-d15687774d7f.png)
![dtt2](https://user-images.githubusercontent.com/93427253/168624155-431682c8-65e4-4217-afa0-3cbb62086ad9.png)
# 1. FUNCTION TRANSFORMATION:
![dtt3](https://user-images.githubusercontent.com/93427253/168624214-6a6e6a42-1216-404d-a66f-472246b1d59a.png)
![dtt4](https://user-images.githubusercontent.com/93427253/168624333-44d5e114-a008-4bc4-8f25-e052902d51ef.png)
![dtt5](https://user-images.githubusercontent.com/93427253/168624377-8f7d94a0-9fed-4727-a2f8-c947378036d9.png)
![dtt6](https://user-images.githubusercontent.com/93427253/168624277-c2552940-141e-426b-b887-a3df01825ac8.png)
# 2. POWER TRANSFORMATION:
![dtt7](https://user-images.githubusercontent.com/93427253/168624960-85cb76bf-e674-488b-8c38-4e37ee02bd1e.png)
![dtt8](https://user-images.githubusercontent.com/93427253/168624985-501a70bf-0388-42f6-8f50-1f0931c19c16.png)
![dtt9](https://user-images.githubusercontent.com/93427253/168625028-e1e7e0ab-95c3-4470-8c97-554532e8d7aa.png)
![dtt10](https://user-images.githubusercontent.com/93427253/168625075-4f990f91-9b89-476e-bd9f-070bcd353308.png)
# 3. QUANTILE TRANSFORAMATION:
![dtt11](https://user-images.githubusercontent.com/93427253/168625181-d00056e9-22db-4f72-bbc7-08a62693e337.png)
![dtt12](https://user-images.githubusercontent.com/93427253/168625211-431e3955-affe-4145-bb87-81668df94417.png)
![dtt13](https://user-images.githubusercontent.com/93427253/168625251-6a91e92e-62f8-4361-9886-df11a54836fe.png)
![dtt14](https://user-images.githubusercontent.com/93427253/168625302-5ac49a55-8079-43f3-abfa-51136c9ac3a9.png)
# FINAL ANALYSATION OF SKEWNESS:
![dtt15](https://user-images.githubusercontent.com/93427253/168625467-10367ecc-235b-46cb-b5fa-e43716a21728.png)

# CODE (Titanic dataset.csv)
```
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
import statsmodels.api as sm  
import scipy.stats as stats  

df=pd.read_csv("titanic_dataset.csv")  
df  

df.drop("Name",axis=1,inplace=True)  
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)  
df.isnull().sum()  

df["Age"]=df["Age"].fillna(df["Age"].median())  
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])  
df.info()  

from sklearn.preprocessing import OrdinalEncoder  
 
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])  
df["Embarked"]=emb.fit_transform(df[["Embarked"]])  

df  

#FUNCTION TRANSFORMATION:  
#Log Transformation  
np.log(df["Fare"])  
#ReciprocalTransformation  
np.reciprocal(df["Age"])  
#Squareroot Transformation:  
np.sqrt(df["Embarked"])  

#POWER TRANSFORMATION:  
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df  
df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])    
df    
df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])  
df  
df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])  
df  
df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])  
df  

#QUANTILE TRANSFORMATION  
from sklearn.preprocessing import QuantileTransformer   
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)  
df["Age_1"]=qt.fit_transform(df[["Age"]])  
sm.qqplot(df['Age'],line='45')  
plt.show()  
sm.qqplot(df['Age_1'],line='45')  
plt.show()  
df["Fare_1"]=qt.fit_transform(df[["Fare"]])  
sm.qqplot(df["Fare"],line='45')  
plt.show()  
sm.qqplot(df['Fare_1'],line='45')  
plt.show()  

df.skew()  
df  
```

# OUPUT
![tit1](https://user-images.githubusercontent.com/93427253/168627509-bf6191d3-91b7-4d38-8ffa-b44547098a46.png)
![tit2](https://user-images.githubusercontent.com/93427253/168627554-341df590-25d0-4930-9dbd-7f48a509d86c.png)
![tit3](https://user-images.githubusercontent.com/93427253/168627694-7dda1a99-38bb-4395-a86a-c714ae21156a.png)
![tit4](https://user-images.githubusercontent.com/93427253/168627737-a956de0b-d261-4da4-a03d-c1ee003411ee.png)
# 1. FUNCTION TRANSFORMATION:
![tit5](https://user-images.githubusercontent.com/93427253/168628006-dc29c59b-9b69-42a6-b755-efa676572883.png)
![tit6](https://user-images.githubusercontent.com/93427253/168628046-6713cb40-0da6-41f3-ae9c-5e12935fb6dd.png)
![tit7](https://user-images.githubusercontent.com/93427253/168628092-9bd555a4-c0c8-469d-918e-68f574a63460.png)
# 2. POWER TRANSFORMATION:
![tit8](https://user-images.githubusercontent.com/93427253/168628453-14f3d094-73f1-49d3-b4f8-5187fc1a13f2.png)
![tit9](https://user-images.githubusercontent.com/93427253/168628483-7931134e-af69-4169-8900-7e593f571c78.png)
![tit10](https://user-images.githubusercontent.com/93427253/168628529-96529af5-343f-4f56-9c77-5d4fd554dfbe.png)
![tit11](https://user-images.githubusercontent.com/93427253/168628558-1a4ffc3b-bef0-4e30-a24b-83d0a9caa3c0.png)
![tit12](https://user-images.githubusercontent.com/93427253/168628609-132256d0-cf92-4f24-9638-9ad614d8033c.png)
# 3. QUANTILE TRANSFORAMATION:
![tit13](https://user-images.githubusercontent.com/93427253/168629028-fd803e4e-c72d-4a0d-a70b-0812afcb60d8.png)
![tit14](https://user-images.githubusercontent.com/93427253/168629082-621b7cf8-aae3-421a-8d67-3e6d695b5f48.png)
# FINAL ANALYSATION OF SKEWNESS:
![tit15](https://user-images.githubusercontent.com/93427253/168629109-ade01997-a830-4569-b979-1b9a0a4cd009.png)
# POINTS:
1. Log transformation is applicable only for right skewed data.
2. Reciprocal transformation is not defined for zeroes.
3. Square transformation is applicable for left skewed data.
4. Quantile transforms are a technique for transforming numerical input or output variables to have a Gaussian or uniform probability distribution.
# RESULT:
Hence, Feature transformation techniques is been performed on given dataset and saved into a file successfully.
