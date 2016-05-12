# -*- coding: utf-8 -*-
# Filename : markovtest.py

from numpy import *

A = [[0.8,0.2],[0.7,0.3]];
A = mat(A);
print A
A1 =A*A
print A1
A10 = A
for i in xrange(9):
	A10 = A10*A
print A10