
# coding: utf-8

# # LEVEL 1 TASK - UNEMPLOYMENT IN INDIA
# 
# Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour force.
# 
#  We have seen a sharp increase in the unemployment rate during Covid-19, so analyzing the unemployment rate can be a good data science project.

# In[1]:


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer


# In[2]:


get_ipython().system('pip install --upgrade scikit-learn')


# **1.Import csv file**

# In[3]:


df=pd.read_csv("Unemployment in India.csv")#import csv files into a dataframe


# **2. Get the informations regarding the data frame**

# In[4]:


df.info()#information regarding the dataframe


# In[5]:


df.head()#first 5 rows


# **3. Get Summary statistics for both numerical and non-numerical columns**

# In[6]:


df.describe(include='all')#Summary statistics for both numerical and non-numerical columns.


# **4. Remove null values from the dataframe**

# In[7]:


df.dropna(inplace=True)# Remove null values


# In[8]:


df.info() 
# 740 Rows


# **5. Identify whether there are missing values in the dataframe**

# In[9]:


missing_values_summary = df.isnull().sum()
missing_values_summary# No missing values


# **6. Identify Outliers in the numerical columns**


# In[10]:


numeric_columns=[' Estimated Unemployment Rate (%)',' Estimated Employed', ' Estimated Labour Participation Rate (%)']
#create a list with all the numeric columns


# In[11]:


plt.figure(figsize=(10, 8))
sns.boxplot(data=df[numeric_columns])
plt.title('Boxplot to determine outliers')
plt.show()


# In[12]:


#Z score = (individual data -mean)/standard deviation
z_scores = ((df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()).abs()


# In[13]:


outliers = (z_scores > 3).any(axis=1) #Rows with z-scores greater than 3 in at least one variable are considered outliers


# In[14]:


print("Rows containing outliers:")
df[outliers]


# **7. Replace outliers with the mean values**

# In[15]:


df[numeric_columns] = np.where(outliers[:, None], df[numeric_columns].mean(), df[numeric_columns])
#np.where - same as if function
#outliers[:, None] - to align outliers with the shape of df[numeric_columns]


# In[16]:


df[outliers]


# **8. Change date column to date specific datatype**
# 
# It will improve performance for date-related operations compared objects.

# In[17]:


df[' Date'] = pd.to_datetime(df[' Date'])
#pd.to_datetime - pandas function to convert and object or string into datetime64 data type


# In[34]:


df.info()


# **9. convert the cleaned data into a csv file**

# In[35]:


df.to_csv('cleaned_data.csv', index=False)


# #### 10. Data Visualization
# 
# **Tableau** is used for data visualization. 
# 

# In[27]:


pip install Pillow


# In[41]:


from PIL import Image

# Path to the JPG file on your local machine
jpg_path = 'Task 1_Unemployment in India - 1_page-0001.jpg'

# Open the JPG file
jpg_image = Image.open(jpg_path)
display(jpg_image)


# As per the analysis, it is found that Haryana has the highest average unemployment rate from May 2019 to May 2020.
# 
# From May 2019 to May 2020 time period, May 2020 has the highest average unemployment rate.
# 

# In[40]:


from PIL import Image

# Path to the JPG file on your local machine
jpg_path = 'Task 1_Unemployment in India - 2_page-0001.jpg'

# Open the JPG file
jpg_image = Image.open(jpg_path)
display(jpg_image)

