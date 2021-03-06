import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.metrics as le_me

import utils



if __name__ == '__main__':
    featFile = './res/feat-hog-paper.npz'


    # load features
    assert (os.path.exists(featFile))
    print('loading ', featFile)
    fileDic = np.load(featFile)
    # set up class names
    usedClasses = ['Strata II - NSC', 'Codman-Hakim', 'Sophysa Polaris - SPV', 'Codman CERTAS', 'Miethke proGAV']


    # Cross validation
    X = fileDic['X']
    y = fileDic['y']
    yPredictionArr, yPredProbArr = utils.crossValidation(X, y)


    # Compute confusion matrix
    confMat = le_me.confusion_matrix(y, yPredictionArr)
    accuracy = (np.trace(confMat)*1.0) /  np.sum( confMat.flatten() )
    print(le_me.classification_report(y, yPredictionArr, target_names=usedClasses))

    # Compute confidence intervals
    utils.bootStrapMetrics(y, yPredictionArr, dataRatio=0.8)

    # Compute statistical significance
    print('-'*20, 'One vs. all tests')
    utils.classSigTests(y, yPredProbArr, usedClasses)

    # # plot confusion matrix
    # plt.figure()
    # utils.plot_confusion_matrix(confMat, usedClasses, normalize=True )
    # plt.show()
