# -*- coding: utf-8 -*-

__author__="alexander philippov"
__date__ ="$06.06.2011 2:36:42$"

import sys
import random
import math

class SOM:
    def __init__(self, inputDim, mapWidth, mapHeight, inputPatternSet):
        self.__eta0   = 0.1
        self.__sigma0 = 2.0
        self.__tau1   = 250
        self.__tau2   = 500

        self.__patternLength = inputDim
        self.__mapWidth      = mapWidth
        self.__mapHeight     = mapHeight

        self.__w = [[[random.random() * 0.1 for i in range(self.__patternLength)]
                                            for i in range(self.__mapWidth)]
                                            for i in range(self.__mapHeight)]

    def coordinates(self, pattern):
        winI = 0
        winJ = 0
        min = sys.float_info.max
        for i in range(self.__mapHeight):
            for j in range(self.__mapWidth):
                s = 0.0
                for k in range(3):
                    s = s + abs(pattern.patternItem[k] - self.__w[i][j][k])
                if min > s:
                    min = s
                    winI = i
                    winJ = j
        return (winI, winJ)

    def clustering(self, patterns):
#        for pattern in patterns:
#            if not isinstance(pattern, IPattern):
#                raise TypeError("Pattern must conform to IPattern")
        #debug
        self.__clu = [[] for i in range(len(patterns))]

        self.__patterns = patterns
        self.__clusteringProcess()

        #self.__writeClu()

    #debug
    def __writeClu(self):
        import csv
        w = csv.writer(file(r'/Users/alexander/temp/data1.csv','wb'))
        w.writerows(self.__clu)
#        for i in range(len(self.__clu)):
#            for j in range(len(self.__clu[i])):
#                print i, j, self.__clu[i][j]

    def __clusteringProcess(self):
        error = 0.0
        e = sys.float_info.max
        de = e - error
        end = False
        i = 0
        while not end:
            error = self.__clusteringEpoch(self.__patterns, i)
            de = abs(e - error)
            if de < 0.0001 or i > 10000:
                end = True
            print "Iteration: ", i
            print "Error: ", e
            print "Error delta: ", de
            e = error
            i = i + 1

    def __clusteringEpoch(self, patterns, n):
        error = 0.0
        randOrder = range(len(patterns))
        for i in range(len(patterns)):
            placeToSwap = random.randint(1, len(patterns) - 1)
            temp = randOrder[i]
            randOrder[i] = randOrder[placeToSwap]
            randOrder[placeToSwap] = temp

        for patIndex in range(len(patterns)):
            currentPatternIndex = randOrder[patIndex]
            pattern = self.__patterns[currentPatternIndex]
            winI = 0
            winJ = 0
            min = sys.float_info.max
            for i in range(self.__mapHeight):
                for j in range(self.__mapWidth):
                    s = 0.0
                    for k in range(3):
                        s = s + pow(pattern.patternItem[k] - self.__w[i][j][k], 2)
                    s = math.sqrt(s)
                    if min > s:
                        min = s
                        winI = i
                        winJ = j
            #self.__clu[currentPatternIndex].append((winI, winJ))
            self.__clu[currentPatternIndex].append(winI * self.__mapWidth + winJ)
            #print "(winI, winJ): ", (winI, winJ)

            self.__printW()

            e = 0.0
            eta = self.__eta(n)
            for i in range(self.__mapHeight):
                for j in range(self.__mapWidth):
                    nbh = self.__neighbourhood(i, j, winI, winJ, n)
                    for k in range(3):
                        dif = self.__patterns[currentPatternIndex].patternItem[k] - self.__w[i][j][k]
                        e = nbh * dif
                        #print "w[", i, "][", j, "][", k, "]: ", self.__w[i][j][k]
                        self.__w[i][j][k] += eta * e;
                        #print "w[", i, "][", j, "][", k, "]: ", self.__w[i][j][k]
                        error += abs(e)

            self.__printW()

        return error

    #for debug
    def __printW(self):
        return
        for i in range(self.__mapHeight):
            for j in range(self.__mapWidth):
                print i * self.__mapWidth + j, " ", self.__w[i][j]

    def __eta(self, n):
        n = n * 1.0
        eta = self.__eta0 * math.exp(-n / self.__tau2)
        #print "Eta(", n, "): ", eta
        return eta

    def __neighbourhood(self, i, j, centerI, centerJ, n):
        di = i - centerI
        dj = j - centerJ
        distance2 = di * di + dj * dj
        
        nbh = math.exp(- distance2 / (2.0 * math.pow(self.__sigma0 * math.exp(-n/self.__tau1), 2)))
        #print "Nbh(", "i: ", i, "j: ", j, "n: ", n, "): ", nbh
        return  nbh

