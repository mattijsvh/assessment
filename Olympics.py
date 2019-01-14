#Packages importeren
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#Inladen data
data = pd.read_csv('athlete_events.csv')
regions = pd.read_csv('noc_regions.csv')

#Mergen van twee dataframes
full_data = pd.merge(data, regions, on = "NOC", how = "left")

#NAN waardes
print(full_data.isnull().sum())

#Vul NAN waardes in Age met gemiddelde
full_data['Age'].fillna((full_data['Age'].mean()), inplace = True)

#Verander NAN waardes in Medal met 'DNW'
full_data['Medal'].fillna('DNW', inplace = True)

#Verander klasse van 'Age'
full_data['Age'] = full_data['Age'].astype('int64')

#Subset van gouden medailles.
gold_medals = full_data[(full_data.Medal == 'Gold')]

#Aantal winnaars van gouden medailles met een leeftijd van > 40
gold_medals['ID'][gold_medals['Age'] > 40].count()

#Maak subset van 'oudere' winnaars van gouden medailles
gold_medals_old = gold_medals['Sport'][gold_medals['Age'] > 40]
gold_medals_old = pd.DataFrame(gold_medals_old)

#Plot van aantal winnaars van gouden medailles met een leeftijd van > 40 per sport
ax0 = sns.countplot(x = 'Sport',
              data = gold_medals_old,
              order = gold_medals_old['Sport'].value_counts().index)
ax0.set_xticklabels(ax0.get_xticklabels(), rotation=90)
ax0.set_title('Aantal gouden medailles per sport van deelnemers die ouder zijn dan 40 jaar')
ax0.set_ylabel('Aantal')
plt.tight_layout()

#Verdeel dataframe in mannen en vrouwen (gebruik alleen zomerspelen)
Men = full_data[(full_data.Sex == 'M') & (full_data.Season == 'Summer')]
Women = full_data[(full_data.Sex == 'F') & (full_data.Season == 'Summer')]

#Aantal deelnemende mannen en vrouwen
Women_sort = Women.groupby('Year')['Sex'].value_counts()
Men_sort = Men.groupby('Year')['Sex'].value_counts()
ax5 = Women_sort.loc[:,'F'].plot()
ax5 = Men_sort.loc[:,'M'].plot()
ax5.set_title('Aantal deelnemende mannen en vrouwen in zomerspelen')
ax5.set_xlabel('Jaar')
ax5.set_ylabel('Aantal')
ax5.legend(['Women', 'Men'])

#Distributie van leeftijden, lengtes en gewicht
plt.figure(figsize=(8,8))

plt.subplot(311)
ax = sns.distplot(full_data['Age'], color = 'blue', kde = True)
ax.set_xlabel('Leeftijd')
ax.set_ylabel('Dichtheid')
ax.set_title('Leeftijdsdistributie', fontsize = 16, fontweight = 200)

plt.subplot(312)
ax1 = sns.distplot(full_data['Height'].dropna(), color = 'Red', kde = True)
ax1.set_xlabel('Lengte')
ax1.set_ylabel('Dichtheid')
ax1.set_title('Lengtedistributie', fontsize = 16, fontweight = 200)

plt.subplot(313)
ax2 = sns.distplot(full_data['Weight'].dropna(), color = 'green', kde = True)
ax2.set_xlabel('Gewicht')
ax2.set_ylabel('Dichtheid')
ax2.set_title('Gewichtsdistributie', fontsize = 16, fontweight = 200)

plt.subplots_adjust(wspace = 0.2, hspace = 0.6, top = 0.9)