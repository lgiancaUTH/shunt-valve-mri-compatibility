import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn.metrics as le_me

import utils



if __name__ == '__main__':
    #-- param
    OTSU = True # set to true to use the threholding enhanced / false for standard


    if OTSU:
        featFile = './res/feat-mrg-CNN-otsu-paper.npz'  # Threshold enhanced features
    else:
        featFile = './res/feat-mrg-CNN-paper.npz' # standard feature
    #--

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
    utils.classSigTests(y, yPredProbArr, usedClasses)


    # Compute confusion matrix
    confMat = le_me.confusion_matrix(y, yPredictionArr)
    accuracy = (np.trace(confMat)*1.0) /  np.sum( confMat.flatten() )

    # Compute confidence intervals
    utils.bootStrapMetrics(y, yPredictionArr, dataRatio=0.8)

    # Compute statistical significance
    print('-'*20, 'One vs. all tests')
    print(le_me.classification_report(y, yPredictionArr, target_names=usedClasses))

    # plot confusion matrix
    plt.figure()
    utils.plot_confusion_matrix(confMat, usedClasses, normalize=True )
    plt.show()
