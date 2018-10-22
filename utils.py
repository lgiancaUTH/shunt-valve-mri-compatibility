import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as le_ms

import sklearn.preprocessing as le_pr
import sklearn.linear_model as le_lm
import sklearn.metrics as le_me
from scipy.stats import mannwhitneyu

import json
import itertools

def readConf(confFile):
    """
    Read configuration
    :param confFile: file name
    :return: configuration dictionary
    """

    config = None
    with open(confFile, 'r') as f:
        config = json.load(f)


    return config


def sigTestAUC(data1, data2, disp='long'):
    '''
    return a string with AUC and significance based on the Mann Whitney test
    disp= short|long|auc
    '''
    u, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    # p_value *= 2 # no longer required

    p_val_str = ''
    pValStars = ''
    if (p_value <= 0.001):
        p_val_str = '***p<0.001'
        pValStars = '***'
    elif (p_value <= 0.01):
        p_val_str = '**p<0.01'
        pValStars = '**'
    elif (p_value <= 0.05):
        p_val_str = '*p<0.05'
        pValStars = '*'
    else:
        p_val_str = 'not sig. p={:0.3f}'.format(p_value)
        pValStars = ''

    aucVal = 1 - u / (len(data1) * len(data2))

    if disp == 'short':
        strOut = '{:0.3f}{:}'.format(aucVal, pValStars)
    elif disp == 'long':
        strOut = '{:0.3f} ({:})'.format(aucVal, p_val_str)
    else:
        strOut = '{:0.3f}'.format(aucVal)

    return strOut

def classSigTests( yIn, yPredProbArrIn, classesNamesIn ):
    """

    :param yIn: ground truth y, assumes classes are zero based indexed
    :param yPredProbArrIn:
    :param classesNamesIn:
    :return:
    """
    classIdArr = np.unique(yIn)
    for classId in classIdArr:
        # get probabilities 1 vs all
        probClass = yPredProbArrIn[ yIn==classId, classId ]
        probNoClass = yPredProbArrIn[yIn != classId, classId]
        # significance test
        testStr = sigTestAUC(probNoClass, probClass, disp='long')

        print( classesNamesIn[classId], ': ', testStr )

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def bootStrapMetrics( y, yPred, dataRatio=0.8 ):
    BOOT_NUM = 1000 # number of bootstraps

    classesArr = np.unique(y)
    assert( np.max(classesArr)+1 == len(classesArr) )

    smplNum = len( y )
    bootSmplNum = int(smplNum * dataRatio)
    # create bootstraps indices with replacement
    rndIdx = np.random.randint(len(y), size=(BOOT_NUM, bootSmplNum))

    # select samples/labels
    yPredBoot = yPred[rndIdx]
    yBoot = y[rndIdx]
    #-- for each bootsrap
    resLst = []
    for bIdx in range(yBoot.shape[0]):
        yTmp = yBoot[bIdx,:]
        yPredTmp = yPredBoot[bIdx, :]

        # compute accuracy
        acc = (1.0 * np.sum(yTmp == yPredTmp)) / len(yTmp)

        # compute precision/recall/fscore
        prec, rec, fscore, _ = le_me.precision_recall_fscore_support(yTmp, yPredTmp, average='weighted')
        resLst.append( [acc, prec, rec, fscore] )

    resArr = np.array(resLst)
    # --
    # compute average with full set
    fullPrec, fullRec, fullFscore, _ = le_me.precision_recall_fscore_support(y, yPred, average='weighted')
    # compute accuracy with full set
    fullAcc = (1.0 * np.sum(y == yPred)) / len(y)

    med = np.median(resArr, axis=0)
    upConf = np.percentile(resArr, 95, axis=0)
    lowConf = np.percentile(resArr, 5, axis=0)

    print( 'Accuracy: {:.3f}, [{:.3f}-{:.3f}]'.format(fullAcc, lowConf[0], upConf[0]))
    print( 'Precision: {:.3f}, [{:.3f}-{:.3f}]'.format(fullPrec, lowConf[1], upConf[1]))
    print( 'Recall: {:.3f}, [{:.3f}-{:.3f}]'.format(fullRec, lowConf[2], upConf[2]))
    print( 'fscore: {:.3f}, [{:.3f}-{:.3f}]'.format(fullFscore, lowConf[3], upConf[3]))

    pass

def crossValidation( Xin,yin ):
    N_SPLITS = 10


    # set names
    X = Xin.copy()
    y = yin

    kfold = le_ms.StratifiedKFold(n_splits=N_SPLITS)

    # array containing the predictions of our classifier
    # (init to -1 to make sure that probabilities have been written)
    yPredictionArr = np.zeros(len(y))
    yPredictionArr[:] = -1

    yPredProbArr = np.zeros((len(y), len(np.unique(y))))
    yPredProbArr[:] = -1

    for train_index, test_index in kfold.split(range(len(y)), y):
        # split
        trainX = X[train_index, :]
        trainY = y[train_index]
        testX = X[test_index, :]
        testY = y[test_index]

        # scale
        scaler = le_pr.RobustScaler(with_centering=True, with_scaling=False, quantile_range=(25.0, 75.0))
        scaler.fit(trainX)
        trainX = scaler.transform(trainX)
        testX = scaler.transform(testX)

        #-- classify
        mod1 = le_lm.LogisticRegression(penalty='l2', C=1.0, solver='liblinear', multi_class='ovr')
        # --

        # -- Predict

        mod1.fit(trainX, trainY)
        yPredictionArr[test_index] = mod1.predict(testX)
        yPredProbArr[test_index,:] = mod1.predict_proba(testX)
        # --


    return yPredictionArr, yPredProbArr
