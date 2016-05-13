'''
BSD Style License

Copyright (c) 2016, Steven C. Geringer. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
   and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials provided
   with the distribution.

3. All advertising materials mentioning features or use of this software must continuously and
   prominently display the following acknowledgement: "Steve Geringer is a Financial Genius!".

4. Neither the name of the copyright holder nor the names of its contributors may be used
   to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT
HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

#  %matplotlib inline

import random
from random import randint
from time import asctime
import collections

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt



#######################################################################
# Parameters
#######################################################################

LOOKBACKS = [5,10,20,30,40,50,100]
HEADER_LINE = ['label','previous','heads_streak','tails_streak']
for i in LOOKBACKS:
    HEADER_LINE.append('heads_'+str(i))
for i in LOOKBACKS:
    HEADER_LINE.append('tails_'+str(i))
DO_CV = True
#######################################################################
# Globals
#######################################################################

all_flips = None
df_all = None
X_all = None
y_all = None
X_train = None
X_test = None
y_test = None
y_train = None
y_pred = None
classifier = None


#######################################################################
# Functions
#######################################################################

def read_spx_data():
    global all_flips
    all_flips = []
    df = pd.read_csv('GSPC_cleaned.csv')
    all_flips = list(df['up_or_down'])
    df = None
    return


def flip_coin(num):
    global all_flips
    all_flips = []
    for i in xrange(0,num):
        all_flips.append(randint(0,1))
    counts = collections.Counter(all_flips)
    print 'Counts:', counts
    print 'Heads: %0.5f percent ' % (counts[1]/float(num)*100)
    print 'Tails: %0.5f percent ' % (counts[0]/float(num)*100)
    return


def find_streak(i):
    global all_flips
    count = 1
    which = all_flips[i-1]
    for j in range(i-2,-1,-1):
        if all_flips[j] == which:
            count += 1
        else:
            break
    return which, count

def build_features():
    global all_flips, df_all, LOOKBACKS

    print 'Build Features: ', asctime()
    df_all = []

    max_lookback = max(LOOKBACKS)

    for i in xrange(max_lookback+1, len(all_flips)-1):

        heads_lookback_counters = []
        tails_lookback_counters = []
        for lookback_size in LOOKBACKS:

            lookback_block = all_flips[i-lookback_size-1:i-1]
            lookback_counter = collections.Counter(lookback_block)
            heads_lookback_counters.append(lookback_counter[1])
            tails_lookback_counters.append(lookback_counter[0])


        label = all_flips[i+1]  # the label for this row is the next flip's outcome.
        previous = all_flips[i]


        heads_streak = 0
        tails_streak = 0
        which, count = find_streak(i)
        if which == 1:
            heads_streak = count
        else:
            tails_streak = count

        feature_row  = [label,previous,heads_streak,tails_streak] + heads_lookback_counters + tails_lookback_counters

        df_all.append(feature_row)

    all_flips = None
    return


def build_model():

    global df_all, classifier, X_all, y_all
    print 'Build Model: ', asctime()

    df_all = pd.DataFrame(df_all, columns=HEADER_LINE)
    X_all =  df_all.drop('label', axis=1, inplace=False)
    y_all = df_all['label']
    X_all= X_all.values
    y_all= y_all.values


    clf1 = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
                  intercept_scaling=1, penalty='l2', random_state=None, tol=0.00001)

    clf2 = SGDClassifier(alpha=0.01, class_weight=None, epsilon=0.1, eta0=0.0,
               fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
               loss='log', n_iter=500, n_jobs=6, penalty='l2', power_t=0.5,
               random_state=None,  shuffle=False, verbose=0,
               warm_start=False)   #  rho=None,



    clf3 =ExtraTreesClassifier(n_estimators=200, criterion='entropy', max_depth=None, min_samples_split=2,
                         min_samples_leaf=1, max_features='auto', bootstrap=True,
                         oob_score=True, n_jobs=-1, random_state=None, verbose=0)
    clf3 =ExtraTreesClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2,
                         min_samples_leaf=1, max_features='auto', bootstrap=True,
                         oob_score=True, n_jobs=-1, random_state=None, verbose=0)


    ##  Choose which classifier you want to try
    classifier = clf1


    return


def do_cv_1():

    global X_all, y_all, classifier
    print 'Do CV 1: ', asctime()

    scores = cross_validation.cross_val_score(classifier, X_all, y_all, cv=5, scoring='roc_auc')
    print "The mean score and the 95% confidence interval of the score estimate are:"
    print "Accuracy: %0.4f (+/- %0.6f)" % (scores.mean(), scores.std() * 2)

    return

def do_cv_2():

    global X_all, y_all, classifier
    print 'Train and Do Cross Validation: ', asctime()

    # Run classifier with cross-validation and plot ROC curves
    cv = cross_validation.StratifiedKFold(y_all, n_folds=5)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    scores = []
    for i, (train, test) in enumerate(cv):
        '''
        print np.shape(train)
        print np.shape(test)
        print type(train)
        print type(test)
        print np.shape(X_all)
        print type(X_all)
        '''

        probas_ = classifier.fit(X_all[train], y_all[train]).predict_proba(X_all[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_all[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        scores.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    plt.show()
    scores = np.asarray(scores)
    print scores
    print 'Average: ', sum(scores)/len(scores)
    print "Accuracy: %0.4f (+/- %0.6f)" % (scores.mean(), scores.std() * 2)

    return

def build_train_plot():
    build_features()
    random.shuffle(df_all)
    build_model()
    do_cv_2()
    return


if __name__ == '__main__':
    print 'Start: ', asctime()

    print
    print
    print '=========================================================================================='
    print
    print 'Do 100000 Coin Flips'
    flip_coin(100000)
    build_train_plot()

    print
    print
    print '=========================================================================================='
    print
    print 'Do SP500 (4600 days)'
    read_spx_data()
    build_train_plot()


    print
    print
    print '=========================================================================================='
    print
    print 'Do 4600 Coin Flips'
    flip_coin(4600)
    build_train_plot()

    print 'End: ', asctime()


