# -*- coding: utf-8 -*-
"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
        注释： 
        这是定义隐藏层的类，首先明确：隐藏层的输入即input，输出即隐藏层的神经元个数。输入层与隐藏层是全连接的。 
        假设输入是n_in维的向量（也可以说时n_in个神经元），隐藏层有n_out个神经元，则因为是全连接， 
        一共有n_in*n_out个权重，故W大小时(n_in,n_out),n_in行n_out列，每一列对应隐藏层的每一个神经元的连接权重。 
        b是偏置，隐藏层有n_out个神经元，故b时n_out维向量。 
        rng即随机数生成器，numpy.random.RandomState，用于初始化W。 
        input训练模型所用到的所有输入，并不是MLP的输入层，MLP的输入层的神经元个数时n_in，而这里的参数input大小是（n_example,n_in）,每一行一个样本，即每一行作为MLP的输入层。 
        activation:激活函数,这里定义为函数tan
        """
        self.input = input  # 类HiddenLayer的input即所传递进来的input 

        """ 
        注释： 
        代码要兼容GPU，则W、b必须使用 dtype=theano.config.floatX,并且定义为theano.shared 
        另外，W的初始化有个规则：如果使用tanh函数，则在-sqrt(6./(n_in+n_hidden))到sqrt(6./(n_in+n_hidden))之间均匀 
        抽取数值来初始化W，若时sigmoid函数，则以上再乘4倍。 
        """  
        #如果W未初始化，则根据上述方法初始化。  
        #加入这个判断的原因是：有时候我们可以用训练好的参数来初始化W，见我的上一篇文章。
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        # 用上面定义的W、b来初始化类HiddenLayer的W、b 
        self.W = W
        self.b = b
        # 隐含层的输出 
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # 隐含层的参数 
        self.params = [self.W, self.b]


# 3层的MLP  
class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        #将隐含层hiddenLayer的输出作为分类层logRegressionLayer的输入，这样就把它们连接了  
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # 以上已经定义好MLP的基本结构，下面是MLP模型的其他参数或者函数 
        
        # 规则化项：常见的L1、L2_sqr  
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # 损失函数Nll（也叫代价函数） 
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        #误差
        self.errors = self.logRegressionLayer.errors

        #MLP的参数
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
# test_mlp是一个应用实例，用梯度下降来优化MLP，针对MNIST数据集
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """ 
    注释： 
    learning_rate学习速率，梯度前的系数。 
    L1_reg、L2_reg：正则化项前的系数，权衡正则化项与Nll项的比重 
    代价函数=Nll+L1_reg*L1或者L2_reg*L2_sqr 
    n_epochs：迭代的最大次数（即训练步数），用于结束优化过程 
    dataset：训练数据的路径 
    n_hidden:隐藏层神经元个数 
    batch_size=20，即每训练完20个样本才计算梯度并更新参数 
    """  
    # 加载数据集，并分为训练集、验证集、测试集。  
    datasets = load_data(dataset)      
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # shape[0]获得行数，一行代表一个样本，故获取的是样本数，除以batch_size可以得到有多少个batch
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    ############
    # 构建模型 #
    ############
    print '... building the model'
    
    index = T.lscalar()  # index表示batch的下标，标量 
    x = T.matrix('x')  # x表示数据集 
    y = T.ivector('y')  # y表示类别，一维向量
    
    rng = numpy.random.RandomState(1234)
    
    #实例化一个MLP，命名为classifier  
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )
    
    #代价函数，有规则化项  
    #用y来初始化，而其实还有一个隐含的参数x在classifier中
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    #cost函数对各个参数的偏导数值，即梯度，存于gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    #参数更新规则  
    #updates[(),(),()....],每个括号里面都是(param, param - learning_rate * gparam)，即每个参数以及它的更新公式
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams) ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ############
    # 训练模型 #
    ############
    print '... training'

    patience = 10000  # 最大迭代次数
    patience_increase = 2  # 步长
    improvement_threshold = 0.995  # 相当大的改善被认为是显著
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # 当前迭代数
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
