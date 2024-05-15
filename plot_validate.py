# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:21:56 2024

@author: NirLavi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

data = {
    "truth" : [],
    "prediction" : []
}

can = pd.read_csv("C:\\Users\\NirLavi\\Desktop\\PhD\\Lab\\IL LAB\\O6Meg\\data\\exp2_randomers\\ROC-AUC\\united\\can_validate_united_full_results.tsv", sep = '\t')
mod = pd.read_csv("C:\\Users\\NirLavi\\Desktop\\PhD\\Lab\\IL LAB\\O6Meg\\data\\exp2_randomers\\ROC-AUC\\united\\mod_validate_united_full_results.tsv", sep = '\t')

#editting can
sub_can = can[["ref_pos", "strand", "mod_probs"]] #choosing the relevant column from the validate file
sub_can = sub_can[sub_can["strand"] == "+"]
sub_can = sub_can.dropna()

can_mod_probs = sub_can["mod_probs"]
can_mod_probs = can_mod_probs[:].str.split(",", expand = True)
can_mod_probs = can_mod_probs[1]
sub_can.insert(0,"truth",0)
sub_can.insert(1,"prediction",can_mod_probs)

sub_can = sub_can[["truth", "prediction", "ref_pos"]]

#editting mod - it will be split into modified and un-mod
sub_mod = mod[["ref_pos", "strand", "mod_probs"]]
sub_mod = sub_mod[sub_mod["strand"] == "+"]
sub_mod = sub_mod.dropna()
mod_mod_probs = sub_mod["mod_probs"]
mod_mod_probs = mod_mod_probs[:].str.split(",", expand = True)
mod_mod_probs = mod_mod_probs[1]
sub_mod.insert(0,"prediction",mod_mod_probs)

sub_mod_modified = sub_mod[sub_mod["ref_pos"] == 21]
sub_mod_modified.insert(0,"truth",1)
sub_mod_modified = sub_mod_modified[["truth", "prediction", "ref_pos"]]


sub_mod_non_modified = sub_mod[sub_mod["ref_pos"] != 21]
sub_mod_non_modified.insert(0,"truth",0)
sub_mod_non_modified = sub_mod_non_modified[["truth", "prediction", "ref_pos"]]

df_mod = pd.concat([sub_mod_non_modified, sub_mod_modified])
df = pd.concat([sub_can, sub_mod_non_modified, sub_mod_modified])

truth = np.array(df.iloc[:,0])
prediction = np.array(df.iloc[:,1], dtype=float)

fpr, tpr, thresholds = metrics.roc_curve(truth, prediction)
auc = metrics.auc(fpr, tpr)
opt_threshold = thresholds[np.argmax(tpr - fpr)]

roc_text = "Optimal Threshold: " + str(round(thresholds[np.argmax(tpr - fpr)],3)) + '\n' + "True Positive Rate: " + str(round(tpr[np.argmax(tpr - fpr)],3))  + "\n" + "False Positive Rate: " + str(round(fpr[np.argmax(tpr - fpr)],3)) + "\n" + "AUC: " + str(round(auc,3))
df['prediction'] = pd.to_numeric(df['prediction'])
df_mod['prediction'] = pd.to_numeric(df_mod['prediction'])
sub_can['prediction'] = pd.to_numeric(sub_can['prediction'])

actual = np.array(df['truth'])
predicted = np.array(df['prediction'] >= opt_threshold)

actual_can = np.array(sub_can['truth'])
predicted_can = np.array(sub_can['prediction'] >= opt_threshold)

confusion_matrix = metrics.confusion_matrix(actual, predicted)
confusion_matrix_can = metrics.confusion_matrix(actual_can, predicted_can)


import matplotlib.patches as mpatches

fig, ax = plt.subplots(6, 1, figsize=(5,30))
fig.tight_layout()

ax[0].plot(fpr, tpr)
ax[0].set_title('ROC Graph')
ax[0].set(xlabel = 'False Positive Rate', ylabel = 'True Positive Rate', label = '(auc=%0.3f)' % auc)
#patch = mpatches.Patch(label='(auc=%0.3f)' % auc)
#ax[0].legend(handles=[patch], loc = "lower right")
ax[0].text(0.6, 0, roc_text, fontsize=8,
          bbox = dict(facecolor = 'white', alpha = 0.5))

plt.subplot(6,1,2)
sns.heatmap(confusion_matrix, annot=True, fmt='g')
ax[1].set(xlabel = 'Prediction', ylabel = 'Truth', title='Confusion Matrix')

sub_can['prediction'] = pd.to_numeric(sub_can['prediction'])
plt.subplot(6,1,3)
sub_can = sub_can.reset_index()
sub_can = sub_can.drop('index', axis=1)
sub_can['passed'] = sub_can['prediction'] >= opt_threshold
for i in range(0, len(sub_can)):
    if sub_can.loc[i, 'truth'] == 1 and sub_can.loc[i, 'passed'] == True:
        sub_can.loc[i, 'class'] = 'TP'
    elif sub_can.loc[i, 'truth'] == 0 and sub_can.loc[i, 'passed'] == True:
        sub_can.loc[i, 'class'] = 'FP'
    elif sub_can.loc[i, 'truth'] == 1 and sub_can.loc[i, 'passed'] == False:
        sub_can.loc[i, 'class'] = 'FN'
    else:
        #mod.loc[i, 'truth'] == 0 and mod.loc[i, 'passed'] == False
        sub_can.loc[i, 'class'] = 'TN'
FPs = sub_can[sub_can['class'] == 'FP']
FPs = FPs.reset_index()
FPs = FPs.sort_values(['ref_pos'])
FPs=FPs.groupby(['ref_pos'],as_index=False).size()
total = sum(FPs['size'])
FPs['size'] = FPs['size']/total
sns.lineplot(data=sub_can, x="ref_pos", y="prediction")

plt.subplot(6,1,3)
ax[2].scatter(sub_can[sub_can['class'] == 'FP']['ref_pos'], sub_can[sub_can['class'] == 'FP']['prediction'],
             c='orange', s=0.1)
ax[2].bar(FPs['ref_pos'], FPs['size'], color = 'orange')
ax[2].hlines(y=opt_threshold, xmin=sub_can.loc[sub_can['ref_pos'].idxmin()]['ref_pos'],
             xmax=sub_can.loc[sub_can['ref_pos'].idxmax()]['ref_pos'],
             linestyles='--', colors = 'red', linewidth = 0.5)
ax[2].set(xlabel = 'Position', ylabel = 'Prediction', title='Canonicals Prediction')

plt.subplot(6,1,4)
sns.heatmap(confusion_matrix_can, annot=True, fmt='g')
ax[3].set(xlabel = 'Prediction', ylabel = 'Truth', title='Canonicals Confusion Matrix')

mod = pd.concat([sub_mod_non_modified, sub_mod_modified])
mod['prediction'] = pd.to_numeric(mod['prediction'])
mod = mod.reset_index()
mod = mod.drop('index', axis=1)
mod['passed'] = mod['prediction'] >= opt_threshold

actual_mod = np.array(mod['truth'])
predicted_mod = np.array(mod['prediction'] >= opt_threshold)
confusion_matrix_mod = metrics.confusion_matrix(actual_mod, predicted_mod)

for i in range(0, len(mod)):
    if mod.loc[i, 'truth'] == 1 and mod.loc[i, 'passed'] == True:
        mod.loc[i, 'class'] = 'TP'
    elif mod.loc[i, 'truth'] == 0 and mod.loc[i, 'passed'] == True:
        mod.loc[i, 'class'] = 'FP'
    elif mod.loc[i, 'truth'] == 1 and mod.loc[i, 'passed'] == False:
        mod.loc[i, 'class'] = 'FN'
    else:
        #mod.loc[i, 'truth'] == 0 and mod.loc[i, 'passed'] == False
        mod.loc[i, 'class'] = 'TN'
FPs = mod[mod['class'] == 'FP']
FPs = FPs.reset_index()
FPs = FPs.sort_values(['ref_pos'])
FPs=FPs.groupby(['ref_pos'],as_index=False).size()
total = sum(FPs['size'])
FPs['size'] = FPs['size']/total

plt.subplot(6,1,5)
sns.lineplot(data=mod, x="ref_pos", y="prediction")
plt.subplot(6,1,5)
ax[4].scatter(mod[mod['class'] == 'FP']['ref_pos'], mod[mod['class'] == 'FP']['prediction'],
             c='orange', s=0.1)
ax[4].bar(FPs['ref_pos'], FPs['size'], color = 'orange')
ax[4].hlines(y=opt_threshold, xmin=mod.loc[mod['ref_pos'].idxmin()]['ref_pos'],
             xmax=mod.loc[mod['ref_pos'].idxmax()]['ref_pos'],
             linestyles='--', colors = 'red', linewidth = 0.5)
ax[4].set(xlabel = 'Position', ylabel = 'Prediction', title='Mod Prediction')

plt.subplot(6,1,6)
sns.heatmap(confusion_matrix_mod, annot=True, fmt='g')
ax[5].set(xlabel = 'Prediction', ylabel = 'Truth', title='Modified Confusion Matrix')


fig.tight_layout()
#plt.subplots_adjust(bottom=1, top=2)
plt.savefig('plot_validate.pdf')
