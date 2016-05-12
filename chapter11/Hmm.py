# -*- coding:utf-8 -*-
# Filename: viterbi.py
from numpy import *

def viterbi(obs, states, start_p, trans_p, emit_p):
	""" 
	:obs:观测序列
	:states:隐状态
	:start_p:初始概率（隐状态）
	:trans_p:转移概率（隐状态）
	:emit_p: 发射概率 （隐状态表现为显状态的概率）
	:return:
	"""
	V = [{}]  # 路径概率表 V[时间][隐状态] = 概率
	for y in states: # 初始化初始状态 (t == 0)
		V[0][y] = start_p[y] * emit_p[y][obs[0]]	
	for t in xrange(1, len(obs)):  # 对 t > 0 跑一遍维特比算法
		V.append({})
		for y in states: 
			# 概率 隐状态 =    前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率 
			V[t][y] = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]) for y0 in states]) 
	result = []
	for vector in V:	
		temp={}
		temp[vector.keys()[argmax(vector.values())]]=max(vector.values())
		result.append(temp)
	return result

#-------------主程序--------------#

states = ('Sunny','Cloudy','Rainy')
obs = ('dry','dryish','soggy')
start_p = {'Sunny':0.63,'Cloudy':0.17,'Rainy':0.20}
trans_p = {
    'Sunny' : {'Sunny': 0.5,'Cloudy':0.375,'Rainy':0.125},
    'Cloudy': {'Sunny': 0.25,'Cloudy':0.125,'Rainy':0.625},
    'Rainy' : {'Sunny': 0.25,'Cloudy':0.375,'Rainy':0.375},
    }
 
emit_p = {
    'Sunny' : {'dry':0.60,'dryish':0.20,'soggy':0.05},
    'Cloudy': {'dry':0.25,'dryish':0.25,'soggy':0.25},   	
    'Rainy' : {'dry':0.05,'dryish':0.10,'soggy':0.50},
}
result = viterbi(obs,states, start_p, trans_p, emit_p)
print result

