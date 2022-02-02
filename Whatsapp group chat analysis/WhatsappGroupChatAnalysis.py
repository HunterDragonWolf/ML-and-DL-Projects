#!/usr/bin/env python
# coding: utf-8

# # WhatsApp Group Chat Analysis
# 

# In[1]:


#Importing the required libraries
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
import itertools 
from collections import Counter
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
#Warnings get ignored
warnings.filterwarnings('ignore')


# In[2]:


#selecting the file and setting formats
file="whatsapp.txt"
key="12hr"
split_formats = {
        '12hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
        '24hr' : '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        'custom' : ''
    }
datetime_formats = {
        '12hr' : '%d/%m/%Y, %I:%M %p - ',
        '24hr' : '%d/%m/%Y, %H:%M - ',
        'custom': ''
    }


# In[3]:


#opening and reading a file
with open(file, 'r', encoding='utf-8') as raw_data:
    # converting the list split by newline char. as one whole string as there can be multi-line messages
    raw_string = ' '.join(raw_data.read().split('\n')) 
    # splits at all the date-time pattern, resulting in list of all the messages with user names
    user_msg = re.split(split_formats[key], raw_string) [1:] 
    # finds all the date-time patterns
    date_time = re.findall(split_formats[key], raw_string) 
    # finds all the date-time patterns
    df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg}) # exporting it to a df


# In[4]:


# converting date-time pattern which is of type String to type datetime,
# format is to be specified for the whole string where the placeholders are extracted by the method 
df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])
    
# split user and msg 
usernames = []
msgs = []
for i in df['user_msg']:
    a = re.split('([\w\W]+?):\s', i) # lazy pattern match to first {user_name}: pattern and spliting it aka each msg from a user
    if(a[1:]): # user typed messages
        usernames.append(a[1])
        msgs.append(a[2])
    else: # other notifications in the group(eg: someone was added, some left ...)
        usernames.append("group_notification")
        msgs.append(a[0])
# creating new columns         
df['user'] = usernames
df['message'] = msgs
# dropping the old user_msg col.
df.drop('user_msg', axis=1, inplace=True)   
df    


# In[5]:


#Checking the info of the df
df.info()


# In[6]:


#Checking the sample of df
df.sample(10)


# In[7]:


#Checking the shape of the message
df[df['message'] == ""].shape[0]


# In[8]:


#Adding extra helper columns for analysis and visualization
df['day'] = df['date_time'].dt.strftime('%a')
df['month'] = df['date_time'].dt.strftime('%b')
df['year'] = df['date_time'].dt.year
df['date'] = df['date_time'].apply(lambda x: x.date())


# In[9]:


df


# In[10]:


#Copying the file rto df1 
df1 = df.copy()      # I will be using a copy of the original data frame everytime, to avoid loss of data!
df1['message_count'] = [1] * df1.shape[0]      # adding extra helper column --> message_count.
df1.drop(columns='year', inplace=True)         # dropping unnecessary columns, using `inplace=True`, since this is copy of the DF and won't affect the original DataFrame.
df1 = df1.groupby('date').sum().reset_index()  # grouping by date; since plot is of frequency of messages --> no. of messages / day.
df1


# In[11]:


#Overall frequency of total messages on the group.

# Improving Default Styles using Seaborn
sns.set_style("darkgrid")
# For better readablity;
import matplotlib
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.figsize'] = (27, 6)      # Same as `plt.figure(figsize = (27, 6))`
# A basic plot
plt.plot(df1.date, df1.message_count,color="orange")
plt.title('Messages sent per day over a time period');
# Could have used Seaborn's lineplot as well.
# sns.lineplot(df1.date, df1.message_count);   
# Saving the plots
plt.savefig('msg_plots.svg', format = 'svg')


# ### Checking the trend for last 10days
# 

# In[12]:


top10days = df1.sort_values(by="message_count", ascending=False).head(10)    # Sort values according to the number of messages per day.
top10days.reset_index(inplace=True)           # reset index in order.
top10days.drop(columns="index", inplace=True) # dropping original indices.
top10days


# In[13]:


#plotting the graph for last 10 days
# Improving Default Styles using Seaborn
sns.set_style("darkgrid")
# For better readablity;
import matplotlib
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.figsize'] = (12, 8)
# A bar plot for top 10 days
sns.barplot(top10days.date, top10days.message_count, palette="hls");
# Saving the plots
plt.savefig('top10_days.svg', format = 'svg')


# # Top 10 active users on the group
# 

# In[14]:


# Total number of people who have sent at least one message on the group;
print(f"Total number of people who have sent at least one message on the group are {len(df.user.unique()) - 1}")   # `-1` because excluding "group_notficiation"
print(f"Number of people who haven't sent even a single message on the group are {237 - len(df.user.unique()) - 1}")


# In[15]:


df2 = df.copy()    
df2 = df2[df2.user != "group_notification"]
top10df = df2.groupby("user")["message"].count().sort_values(ascending=False)

# Final Data Frame
top10df = top10df.head(10).reset_index()
top10df


# In[16]:


top10df['initials'] = ''
for i in range(10):
    top10df.initials[i] = top10df.user[i].split()[0][0] + top10df.user[i].split()[1][0]
top10df.initials[7] = "Me"    # That's me
top10df.initials[8] = "DT"


# In[17]:


# For better readablity;
import matplotlib
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[18]:


# Beautifying Default Styles using Seaborn
sns.set_style("darkgrid")
sns.barplot(top10df.initials, top10df.message, data=top10df);
top10df.initials


# ### Most used words in the chat
# 

# In[20]:


comment_words = ' '
stopwords = STOPWORDS.update(['group', 'link', 'invite', 'joined', 'message', 'deleted', 'yeah', 'hai', 'yes', 'okay', 'ok', 'will', 'use', 'using', 'one', 'know', 'guy', 'group', 'media', 'omitted'])
# iterate through the DataFrame.
for val in df2.message.values:
    # typecaste each val to string.
    val = str(val) 
    # split the value.
    tokens = val.split() 
    # Converts each token into lowercase.
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    for words in tokens: 
        comment_words = comment_words + words + ' '
wordcloud = WordCloud(width = 600, height = 600, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 8).generate(comment_words)


# In[21]:


wordcloud.to_image()

