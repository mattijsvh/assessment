import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random
from bokeh.io import output_file, show
from bokeh.models import HoverTool
from bokeh.plotting import figure

#plt.plot
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)
plt.plot(t, s)
plt.title('Voltage per seconde')
plt.xlabel('Tijd (s)')
plt.ylabel('Voltage (mV)')

#plt.hist
x = np.random.normal(size = 1000)
plt.hist(x, normed=True, bins=30)
plt.title('Normaaldistributie van 1000 willekeurige waardes')
plt.ylabel('Waarschijnlijkheid')

#plt.bar
men = (85, 75, 96, 121, 59)
women = (74, 72, 85, 101, 57)

fig, ax = plt.subplots()
index = np.arange(5)
bar_width = 0.35

a = plt.bar(index, men, bar_width,
                 color = 'b',
                 label = 'men')

b = plt.bar(index + bar_width, women, bar_width,
                 color = 'g',
                 label = 'women')

plt.xlabel('Groep')
plt.ylabel('Scores')
plt.title('Scores per groep')
plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
plt.legend()

plt.tight_layout()
plt.show()

#sns.kdeplot (kernel density estimation)
d = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
d = pd.DataFrame(d, columns=['x', 'y'])

for col in 'xy':
    sns.kdeplot(d[col], shade=True)
plt.title('Voorspelling van distributie')

#sns.lmplot
df = pd.DataFrame()

df['x'] = random.sample(range(1, 1000), 100)
df['y'] = random.sample(range(1, 1000), 100)

sns.set_style("ticks")
sns.lmplot(x= 'x',
           y = 'y',
           data=df)

#Bokeh hexbin plot
a1 = 2 + 2*np.random.standard_normal(500)
a2 = 2 + 2*np.random.standard_normal(500)

p = figure(title = "Hexbin met 500 punten",
           match_aspect = True,
           tools = "wheel_zoom,reset",
           background_fill_color = '#5f5c60')
p.grid.visible = False

r, bins = p.hexbin(a1, a2, size=0.5, hover_color="white", hover_alpha=0.6)

p.circle(a1, a2, color="aqua", size=1)

p.add_tools(
    HoverTool(
        tooltips = [("Qty", "@c")],
        mode = "mouse",
        point_policy = "follow_mouse",
        renderers = [r]
))

output_file("hexbin.html")

show(p)