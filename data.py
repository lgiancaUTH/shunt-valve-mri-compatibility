import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import json
import os
import skimage.io as skio

class ValveDataset:

    def __init__(self, confIn):
        """
        Constructor for valve dataset
        :param confIn: configuration
        """
        self._conf = confIn
        self._classArr = []
        self._gtFr = None

        dirLst = os.listdir(self._conf['data_dir'])
        # create class IDs
        for dirTmp in dirLst:
            if dirTmp[0] == '_':
                continue
            elif dirTmp[0] == '.':
                continue
            #else
            self._classArr.append(dirTmp)

        # generate file lst
        fileDic = {'fileName': [], 'classID': np.array([], dtype=int) }
        for classId in range(len(self._classArr)):

            # get files for a given directory (which correspond to a given class classId)
            fullImgDir = confIn['data_dir']  + self._classArr[classId]
            imgFiles = os.listdir( fullImgDir )
            # add subdirectory
            imgFiles = [ (self._classArr[classId] + '/' + imgFile) for imgFile in imgFiles ]
            fileDic['fileName'] = np.append(fileDic['fileName'], imgFiles)
            # add classIDs, as many as images
            fileDic['classID'] = np.append(fileDic['classID'], np.ones(len(imgFiles), dtype=int)* classId)

        # create dataframe with full dataset and labels
        self._gtFr = pd.DataFrame(fileDic)
        # add labels description
        self._gtFr['classStr'] =   self._gtFr['classID'].apply( lambda x: self._classArr[x] )
        # remove database files
        self._gtFr = self._gtFr[~ self._gtFr.fileName.str.endswith('.db')]

    def loadInfo(self, imgID):
        """
        Load information
        :param imgID: numerical id
        :return: information about image as dictionary
        """
        return self._gtFr.loc[imgID].to_dict()

    def loadImg(self, imgID):
        """
        load image
        :param imgID: numerical id
        :return: (info, image)
        """
        info = self.loadInfo(imgID)

        return info, skio.imread(self._conf['data_dir'] + info['fileName'])

    def viewImg(self, imgID):
        """
        Display image
        :param imgID: numerical id
        :return: (info, image)
        """

        _, img = self.loadImg(imgID)

        plt.figure()
        plt.imshow( img )
        plt.show()


if __name__ == '__main__':
    CONF_FILE = '../res/configurationGpu.json'
    CONF = utils.readConf(CONF_FILE)
    valveSet = ValveDataset(CONF)

    valveSet = ValveDataset(CONF)
    valveSet.viewImg(100)