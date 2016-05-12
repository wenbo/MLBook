# -*- coding: utf-8 -*-

__author__="alexander@philippov.su"
__date__ ="$05.06.2011 23:58:41$"

from abc import ABCMeta, abstractmethod, abstractproperty

class ImagePattern():
    patternLength = 3
    def __init__(self, imageFileName):
        self.patternItem = list()
        self.__patternLength = ImagePattern.patternLength
        self.__imageFileName = imageFileName
        self.__clusterCoordinates = (0, 0)
        self.analyze()

    @property
    def patternLength(self):
        return self.__patternLength

    @property
    def clusterCoordinates(self):
        return self.__clusterCoordinates

    @clusterCoordinates.setter
    def clusterCoordinates(self, value):
        if not isinstance(value, tuple):
            raise TypeError("Coordinates must be a tuple of numbers")
        self.__clusterCoordinates = value

    def analyze(self):
        from PIL import Image
        
        im = Image.open(self.__imageFileName)
        if im.mode != "RGB":
            self.patternItem = [0, 0, 0]
            return
        try:
            histo = im.histogram()
        except:
            self.patternItem = [0, 0, 0]
            return
        sr = 0.0
        sg = 0.0
        sb = 0.0
        for i in range(255):
            sr = sr + histo[i +   0] * i
            sg = sg + histo[i + 256] * i
            sb = sb + histo[i + 512] * i
        print sr
        print sg
        print sb
        ims = im.size[0] * im.size[1]
        r = sr / ims / 256.0
        g = sg / ims / 256.0
        b = sb / ims / 256.0
        print "r: ", r
        print "g: ", g
        print "b: ", b

        self.patternItem = [r, g, b]

class ImageContainer(object):
    def __init__(self):
        self.__dirPath = ""
        self.__images = list()

    @property
    def images(self):
        return self.__images;

    def addImage(self, img):
        self.__images.append(img)

    def __processFolder(self,):
        import os
        import os.path
        #files = os.listdir(self.__dirPath) #для только одной папки
        c = 0
        for root, dirs, files in os.walk(self.__dirPath):
            for file in files:
                if file.find("jpg") > 0:
                    print "Adding file: ", file
                    img = ImagePattern(os.path.join(root, file))
                    self.addImage(img)
                    c = c + 1
        print "Added ", c, " images"

    def fromDirectory(self, dirPath):
        self.__dirPath = dirPath
        self.__images = list()

        self.__processFolder()

    def analyzeAll(self):
        for img in self.__images:
            img.analyze()
            