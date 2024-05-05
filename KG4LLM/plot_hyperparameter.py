from matplotlib import pyplot as plt
plt.style.use('seaborn')
import seaborn
# seaborn.set(rc={'axes.facecolor':'gainsboro'})

color1 = "#808080"
color2 = "#038355"
color3 = '#9BB8F2'
color4 = "#ec661f"
fig, ax1 = plt.subplots()
ax1.set_ylim(50,86)
k = [1,3,5,10,20,30]
scores_avg = [68.6, 72.0, 72.6, 70.2, 71.6, 71.1]
scores_medmc = [ 71.4,71.9,75.2,71.4,72.9,70.4]
scores_med = [ 57.9,58.6,57.9,61.8,57.9,57.9]
scores_qa4mre = [ 71.4, 80.0, 71.4, 74.3, 74.3, 74.3]

ax1.tick_params(axis='y', labelsize=16)
ax1.tick_params(axis='x', labelsize=16)
ax1.plot(k,scores_med, color=color2, marker='s', label='MedQA')
ax1.plot(k,scores_medmc, color=color3, marker='v', label='MedMCQA')
ax1.plot(k,scores_qa4mre, color=color4, marker='*', label='QA4MRE')
ax1.plot(k,scores_avg, color=color1, marker='o', label='AVG')
ax1.legend(loc='upper right', prop = {'size':16})
ax1.set_xlabel('k', size=20)
ax1.set_ylabel('Accuracy (%)', size=20)

plt.show()
plt.savefig('hyper-parameter.png')