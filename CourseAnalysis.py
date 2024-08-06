import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string, re, nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#nltk.download('stopwords')
from datetime import datetime
import plotly.express as px
import warnings
#load the four diferent datasets
df_biz = pd.read_csv("C:\\Users\\SATWIK SAHOO\\PycharmProjects\\pythonProject1\\Business Courses.csv") #Business Finance
df_gfx = pd.read_csv("C:\\Users\\SATWIK SAHOO\\PycharmProjects\\pythonProject1\\Design Courses.csv") #Graphics
df_mus = pd.read_csv("C:\\Users\\SATWIK SAHOO\\PycharmProjects\\pythonProject1\\Music Courses.csv") #Musical Instrument
df_dev = pd.read_csv("C:\\Users\\SATWIK SAHOO\\PycharmProjects\\pythonProject1\\Web development Courses.csv") #Web Development

#join the different dataframes
df = pd.concat([df_biz, df_gfx, df_mus, df_dev])

#sample the first rows of the dataframe
print('\n')
print("This combined dataset has {} rows and {} columns".format(df.shape[0], df.shape[1]) )
df.head(2)
#summary statistics of the combined dataframe
df.describe()
#check data types and missing values
df.info()
#check for and drop duplicates
df[df.duplicated()]

#drop duplicates in the course id column
df.drop_duplicates(subset=['course_id'],inplace=True)
#drop rows with missing values
df.dropna(inplace=True)

#confirm there is no more any null values
df.isnull().sum()
#drop unwanted columns from the combined dataframe
df = df.drop(['url','num_reviews', 'course_id', 'num_lectures','content_duration'], axis = 1)
#cast datatype as int
df['num_subscribers'] = df['num_subscribers'].astype('int64')


#confirm change
df.head(2)
#categorize courses as either free or paid
conditions = [
    (df['price'] == 0), 
    (df['price'] > 0)]

values  = ['free', 'paid']
df['price_group'] = np.select(conditions, values)

df['price_group'].value_counts()
#set category for ratings using a 5star system
conditions1 = [
    (df['Rating'] == 0),
    (df['Rating'] > 0) & (df['Rating'] < 0.2),
    (df['Rating'] >= 0.2) & (df['Rating'] < 0.4),
    (df['Rating'] >= 0.4) & (df['Rating'] < 0.6),
    (df['Rating'] >= 0.6) & (df['Rating'] < 0.8),
    (df['Rating'] >= 0.8) 
]
values1 = [0, 1, 2, 3, 4, 5]
df['star_rating'] = np.select(conditions1, values1)

df['star_rating'].value_counts()
#rename the levels
levels = ({'All Levels':'General', 'Beginner Level':'Beginner', 
           'Intermediate Level':'Intermediate', 'Expert Level':'Expert'})
df['level'] = df['level'].replace(levels)

df['level'].value_counts()
#extract date only from datetime object
df['published'] = df['published_timestamp'].str.split('T').str[0]

df = df.drop(['published_timestamp'], axis = 1) #drop the combined timestam column

df['published']
#create column for the year the course was published
df['year'] = (df['published'].str.split('-').str[0]).astype(int)
df['year'].value_counts()
def clean(text):
    text = str(text).title()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text) #remove url
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) #remove punctuations
    text = re.sub('\n', '', text)
    text = re.sub('[0-9]', '', text)
    text = re.sub('<.*?>+', '', text)
    return text

df['course_title'] = df['course_title'].apply(clean)
#clean off punctautions from subject column

df['subject']=df['subject'].str.split(': ').str[-1].str.lstrip()
df['subject'].value_counts()
df['revenue'] = df['num_subscribers'] * df['price']
#separate the component dataframes

df_biz = df.query('subject == "Business Finance"')
df_gfx = df.query('subject == "Graphic Design"')
df_mus = df.query('subject == "Musical Instruments"')
df_dev = df.query('subject == "Web Development"')

print('There are {} {} courses'.format(df_biz.shape[0], df_biz.subject[0]))
print('There are {} {} courses'.format(df_gfx.shape[0], df_gfx.subject[0]))
print('There are {} {} courses'.format(df_mus.shape[0], df_mus.subject[0]))
print('There are {} {} courses'.format(df_dev.shape[0], df_dev.subject[0]))
#plot the course category

#extract only the subject name
df['subject'] = df['subject'].str.split(': ').str[-1]
df['subject'] = df['subject'].str.lstrip(' ')

subject = df['subject'].value_counts()

print(df['subject'].value_counts())
figure = px.pie(df, 
                values = subject.values,
                names = subject.index,
                hole = 0.5)
figure.show()
#course price versus number of subscribers
fig, ((ax0, ax1), (ax2, ax3 )) = plt.subplots(2,2 ,figsize=(15,10))

values = df.groupby('subject')['price'].mean().round()
ax0.bar(df.groupby('subject')['price'].mean().index, df.groupby('subject')['price'].mean() )
ax0.set_title('Average Course Price')
[ax0.text(index, value, str(value)) for index, value in enumerate(values)] #list comprehension to add bar values
    
values = df.groupby('subject')['num_subscribers'].mean().round()    
ax1.bar(df.groupby('subject')['num_subscribers'].sum().index, df.groupby('subject')['num_subscribers'].sum())
ax1.set_title('Number of Subscribers')
[ax1.text(index, value, str(value)) for index, value in enumerate(values)]

values = df.groupby('subject')['course_title'].count()
ax2.bar(df.groupby('subject')['course_title'].count().index, df.groupby('subject')['course_title'].count())
ax2.set_title('Number of Published Courses')
[ax2.text(index, value, str(value)) for index, value in enumerate(values)]

values = df.groupby('subject')['Rating'].mean().round(2)
ax3.bar(df.groupby('subject')['Rating'].mean().index, df.groupby('subject')['Rating'].mean())
ax3.set_title('Average Rating')
[ax3.text(index, value, str(value)) for index, value in enumerate(values)]

plt.tight_layout()
plt.show()
#revenue generation by subject and by course
fig, ((ax0,ax1), (ax2,ax3)) = plt.subplots(2,2 ,figsize=(15,10))

df_free = df.query('price == 0')
(ax0.barh(df_free.groupby('subject')['course_title'].count().index, 
          100*df_free.groupby('subject')['course_title'].count()/
          df.groupby('subject')['course_title'].count()))
ax0.set_title('% Free Courses per Subject')
ax0.set_xlabel('Number of Courses')

df_paid = df.query('price != 0')
ax1.barh(df_paid.groupby('subject')['course_title'].count().index, df_paid.groupby('subject')['course_title'].count())
ax1.set_title('Paid Courses per Subject')
ax1.set_xlabel('Number of Courses')

ax2.barh(df.groupby('level')['price'].mean().index, df.groupby('level')['price'].mean())
ax2.set_title('Average Price for Different Levels of Learning')
ax2.set_xlabel('Price')

df_free = df.query('price == 0')
(ax3.barh(df_free.groupby('level')['course_title'].count().index, 
          100*df_free.groupby('level')['course_title'].count()/
          df.groupby('level')['course_title'].count()))
ax3.set_title('% Free Courses Across Levels')
ax3.set_xlabel('NUmber of Courses')

plt.tight_layout()
plt.show()
#revenue generation by subject and by course
fig, ((ax0), (ax1)) = plt.subplots(1, 2 ,figsize=(15,10))

(ax0.barh(df.groupby('subject')['revenue'].sum().index, sorted(df.groupby('subject')['revenue'].mean())))
ax0.set_title('Revenue Per Subject')
ax0.set_xlabel('Revenue')


#sort the dataframe by revenue generated and courses and plot top 15
df_rev = df[['course_title', 'revenue']].sort_values('revenue', ascending=False).head(15)
(ax1.barh(df_rev.head(15).groupby('course_title')['revenue'].sum().index, 
          sorted(df_rev.head(15).groupby('course_title')['revenue'].sum())))
ax1.set_title('Top 10 Revenue Generating Courses')
ax1.set_xlabel('Revenue')

plt.tight_layout()
plt.show()
#plot of the courses by number of subscribers
df_sub = df.sort_values('num_subscribers', ascending=False).head(20)
df_sub.groupby(['course_title', 'price_group','subject'])['num_subscribers'].sum().sort_values().plot(kind='barh')
plt.title('20 Most Subscribed Courses')
plt.ylabel('')
plt.xlabel('Number of Subscribers')
plt.rcParams["figure.figsize"] = (10, 12)
plt.show()
#distribution chart of selected attributes
fig = plt.figure(figsize = (12,5))
ax = fig.gca()

df[['price', 'Rating']].hist(ax = ax)
plt.tight_layout()
plt.xticks(rotation=45, ha='right')
warnings.filterwarnings("ignore")

plt.show()
df.plot.hexbin(x='price', y='Rating', gridsize=30)
plt.show()
#average price of course per category
print(df.groupby('subject')['price'].mean())

#use power query to categorize dataset based on the average price
below_biz = df_biz.query('price < 68.694374')
above_biz = df_biz.query('price >= 68.694374')

below_gfx = df_gfx.query('price < 57.890365')
above_gfx = df_gfx.query('price >= 57.890365')

below_mus = df_mus.query('price < 49.558824')
above_mus = df_mus.query('price >= 49.558824')

below_dev = df_dev.query('price < 77.036575')
above_dev = df_dev.query('price >= 77.036575')

#use this to compare number of courses, rating etc below and above the median(or average) price
#double hist plot function

fig, ((ax0, ax1), (ax2, ax3 )) = plt.subplots(2,2 ,figsize=(15,10))

ax0.hist(below_biz.num_subscribers, 5, color ='r', alpha = 0.5, label='Below Average Price')
ax0.hist(above_biz.num_subscribers, 5, color ='b', alpha = 0.5, label='Above Average Price')
ax0.legend(prop={'size': 10})
ax0.set_xlabel('Number of Subscribers')
ax0.set_ylabel('Count')
ax0.set_title('Business Finance')

ax1.hist(below_gfx.num_subscribers, 5, color ='r', alpha = 0.5, label='Below Average Price')
ax1.hist(above_gfx.num_subscribers, 5, color ='b', alpha = 0.5, label='Above Average Prices')
ax1.legend(prop={'size': 10})
ax1.set_xlabel('Number of Subscribers')
ax1.set_title('Graphics Design')

ax2.hist(below_mus.num_subscribers, 5, color ='r', alpha = 0.5, label='Below Average Price')
ax2.hist(above_mus.num_subscribers,5, color ='b', alpha = 0.5, label='Above Average Price')
ax2.legend(prop={'size': 10})
ax2.set_xlabel('Number of Subscribers')
ax2.set_ylabel('Count')
ax2.set_title('Musical Instrument')

ax3.hist(below_dev.num_subscribers, 5, color ='r', alpha = 0.5, label='Below Average Price')
ax3.hist(above_mus.num_subscribers,5, color ='b', alpha = 0.5, label='Above Average Price')
ax3.legend(prop={'size': 10})
ax3.set_xlabel('Number of Subscribers')
ax3.set_title('Web Development')

fig.tight_layout()
plt.show()
#course price versus number of subscribers
fig, ((ax0, ax1), (ax2, ax3 )) = plt.subplots(2,2 ,figsize=(15,10))

ax0.bar(df_biz.price, df_biz['num_subscribers'], width=2.4)
ax0.set_title('Business Finance')
ax0.set_ylabel('Number of Subscribers')

ax1.bar(df_gfx.price, df_gfx.num_subscribers, width=2.4)
ax1.set_title('Graphics Design')

ax2.bar(df_mus.price, df_mus.num_subscribers, width=2.4)
ax2.set_title('Musical Instrument')
ax2.set_xlabel('Course Price')
ax2.set_ylabel('Number of Subscribers')

ax3.bar(df_dev.price, df_dev.num_subscribers, width=2.4)
ax3.set_title('Web Development')
ax3.set_xlabel('Course Price')

plt.show()
#pieplot for the different courses
# use 5 for number of bins
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2 ,figsize=(15,10))

labels = [0,1,2,3,4,5]

ax0.pie(df_biz['star_rating'].value_counts(), autopct='%1.1f%%',  wedgeprops=dict(width=.4))
ax0.legend(df_biz['star_rating'].value_counts().index,title = '5-star ratings')
ax0.set_title('Business Finance')

ax1.pie(df_gfx['star_rating'].value_counts(), autopct='%1.1f%%',   wedgeprops=dict(width=.4))
ax1.legend(df_biz['star_rating'].value_counts().index,title = '5-star ratings')
ax1.set_title('Graphic Design')

ax2.pie(df_mus['star_rating'].value_counts(), autopct='%1.1f%%',  wedgeprops=dict(width=.4))
ax2.legend(df_biz['star_rating'].value_counts().index,title = '5-star ratings')
ax2.set_title('Musical Instruments')

ax3.pie(df_dev['star_rating'].value_counts(), autopct='%1.1f%%',  wedgeprops=dict(width=.4))
ax3.legend(df_biz['star_rating'].value_counts().index,title = '5-star ratings')
ax3.set_title('Web Development')

fig.tight_layout()
plt.show()
fig, ((ax0), (ax1)) = plt.subplots(1,2 ,figsize=(15,10))

#paid courses with above average subscription
subject = df[(df['num_subscribers']>3199) & (df['price_group'] != 'free')]['level'].value_counts()
ax0.pie(subject, autopct='%1.1f%%',   wedgeprops=dict(width=.4))
ax0.legend(df['level'].value_counts().index, title = 'Level')
ax0.set_title('Levels of Paid Courses with above Average Subscription Count')

#paid courses with above average subscription
subject1 = df[(df['num_subscribers']>3199) & (df['price_group'] != 'free')]['subject'].value_counts()
ax1.pie(subject1, autopct='%1.1f%%',   wedgeprops=dict(width=.4))
ax1.legend(df['subject'].value_counts().index, title = 'Subject')
ax1.set_title('Category of Paid Courses with above Average Subscription Count')

plt.show()
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2 ,figsize=(15,10))

labels = [0,1,2,3,4,5]

ax0.bar(df_biz.year.value_counts().index,df_biz.year.value_counts().values)
ax0.set_title('Business Finance')

ax1.bar(df_gfx.year.value_counts().index,df_gfx.year.value_counts().values,color='r')
ax1.set_title('Graphic Design')

ax2.bar(df_mus.year.value_counts().index,df_mus.year.value_counts().values,color='y')
ax2.set_title('Musical Instruments')

ax3.bar(df_dev.year.value_counts().index,df_dev.year.value_counts().values, color='c')
ax3.set_title('Web Development')

fig.tight_layout()
plt.show()
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2 ,figsize=(15,10))

labels = [0,1,2,3,4,5]

ax0.bar(df_biz.groupby('year')['revenue'].sum().index, df_biz.groupby('year')['revenue'].sum())
ax0.set_title('Business Finance')

ax1.bar(df_gfx.groupby('year')['revenue'].sum().index, df_gfx.groupby('year')['revenue'].sum(), color='r')
ax1.set_title('Graphic Design')

ax2.bar(df_mus.groupby('year')['revenue'].sum().index, df_mus.groupby('year')['revenue'].sum(),color='y')
ax2.set_title('Musical Instruments')

ax3.bar(df_dev.groupby('year')['revenue'].sum().index, df_dev.groupby('year')['revenue'].sum(), color='c')
ax3.set_title('Web Development')

fig.tight_layout()
plt.show()