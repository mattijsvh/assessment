import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

df = pd.DataFrame()
df['x'] = random.sample(range(1, 1000), 100)
df['y'] = random.sample(range(1, 1000), 100)

#Berekenen van gemiddelde
np.mean(df) #Som van alle waarden gedeeld door het aantal waarden
m1 = sum(df['x']) / len(df['x'])
m2 = sum(df['y']) / len(df['y'])

#Berekenen van variantie
np.var(df) #Variantie van waarde in vergelijking met het gemiddelde
var1 = sum((xi - m1) ** 2 for xi in df['x']) / len(df['x'])
var2 = sum((xi - m2) ** 2 for xi in df['y']) / len(df['y'])

#Berekenen van standaarddeviatie
np.std(df) #Spreiding rondom het gemiddelde (wortel van variantie)
np.sqrt(var1)
np.sqrt(var2)

#Spreiding en uitschieters
spreiding = np.random.rand(50) * 100
centrum = np.ones(25) * 50
uitschieter_hoog = np.random.rand(10) * 100 + 100
uitschieter_laag = np.random.rand(10) * -100
data = np.concatenate((spreiding, centrum, uitschieter_hoog, uitschieter_laag))

fig1, ax1 = plt.subplots()
ax1.set_title('Boxplot')
ax1.boxplot(data, flierprops = dict(markerfacecolor='y', marker='D'))
ax1.set_xticks(())

#Lineaire regressie toepassen op diabetes dataset
diabetes = datasets.load_diabetes()

# Gebruik 1 kolom van diabetes dataset
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Verdeel de data in training en test datasets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Maak een object voor lineaire regressie: regr
regr = linear_model.LinearRegression()

# Train het model door beide train-datasets te gebruiken
regr.fit(diabetes_X_train, diabetes_y_train)

# Maak voorspellingen op basis van de test-dataset
diabetes_y_pred = regr.predict(diabetes_X_test)

# Plot de uitkomst
plt.scatter(diabetes_X_test,
            diabetes_y_test,
            color = 'black')
plt.plot(diabetes_X_test,
         diabetes_y_pred,
         color = 'purple',
         linewidth = 2)
plt.xticks(())
plt.yticks(())
plt.show()