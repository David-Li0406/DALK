from matplotlib import pyplot as plt
plt.style.use('seaborn')
import seaborn
# seaborn.set(rc={'axes.facecolor':'gainsboro'})

color1 = "#ec661f"
color2 = "#038355"
color3 = '#9BB8F2'

years = [2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
ours = [68.1,69.0,70.2,68.4,69.7,70.3,70.3,71.8,71.2,72.6]
no_self = [69.0,69.4,70.0,70.0,70.1,69.9,70.2,68.0,69.8,70.6]
baseline = [67.1 for _ in years]

count = [5661,6953,6349,7136,6587,5365,4953,5071,3309,3201]
count = [sum(count[:i+1]) for i in range(len(count))]

fig, ax2 = plt.subplots()
ax1 = ax2.twinx()
ax1.set_ylim(66,76)
ax1.plot(years,ours, color=color1, marker='o', label='DALK')
ax1.plot(years,no_self, color=color2, marker='v', label='w/o self-aware knowledge retrieval')
ax1.plot(years,baseline, color='blue', linestyle='--', label='Baseline')
ax2.bar(years, count,color=color3)


ax1.tick_params(axis='y', labelsize=16)
ax2.tick_params(axis='x', labelsize=16)

ax2.tick_params(axis='y', labelsize=16)

ax1.set_xlabel('Year', size=16)
ax1.set_ylabel('Accuracy (%)', size=16)
ax2.set_ylabel('Triplet Number (#)', size=16)

ax1.legend(loc='upper left', prop = {'size':16})
plt.show()
plt.savefig('evolution.png')
