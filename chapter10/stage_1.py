# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

# 定义矩阵与矩阵运算
X = T.matrix("X")
results, updates = theano.scan(lambda x_i: T.sqrt((x_i ** 2).sum()), sequences=[X])
compute_norm_lines = theano.function(inputs=[X], outputs=[results])

# np.diag:对角阵
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)
print compute_norm_lines(x)[0]

# comparison with numpy
print np.sqrt((x ** 2).sum(1))