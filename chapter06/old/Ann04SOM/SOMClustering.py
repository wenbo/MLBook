# -*- coding: utf-8 -*-

__author__="alexander"
__date__ ="$05.06.2011 21:30:11$"

from ImageContainer import ImageContainer
from ImageContainer import ImagePattern

from SOM import SOM

if __name__ == "__main__":
    imgContainer = ImageContainer()
    imgContainer.fromDirectory("/Users/alexander/Pictures/qwe/")
    #imgContainer.analyzeAll()
    som = SOM(3, 5, 5, None)
    som.clustering(imgContainer.images)

    for image in imgContainer.images:
        print som.coordinates(image)