# -*- coding: utf-8 -*-

"""
本教程介绍了使用Theano和随机梯度下降logistic回归。Logistic回归是一个概率的，线性分类器。
它是一个参数化的权重矩阵:"W"和一个偏移向量'b'。
分类器是通过投影数据点到一组超平面的完成，距离被用于确定一个类别成员概率。
它被写为:
本教程介绍了适用于大型数据集随机梯度下降优化方法。

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time
import numpy
import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):

        # 初始化权重W为全0矩阵 shape(n_in, n_out) W 是一个矩阵，第k列表示为第k类的分隔超平面
        self.W = theano.shared(
            value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX ),
            name='W',
            borrow=True
        )
        # 初始化偏移量b为全0向量，n_out 0; b 是个向量，其中元素k表示为超平面k的自由参数
        self.b = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX ),
            name='b',
            borrow=True
        )

        # 计算类成员概率的符号表达式矩阵，其中：
        # x:input是个矩阵，其中 row-j 表示为第j个输入训练样本
        # input是(n_example,n_in)，W是（n_in,n_out）,点乘得到(n_example,n_out)，加上偏置b，  
        # 再作为T.nnet.softmax的输入，得到p_y_given_x  
        # 故p_y_given_x每一行代表每一个样本被估计为各类别的概率      
        # PS：b是n_out维向量，与(n_example,n_out)矩阵相加，内部其实是先复制n_example个b，  
        # 然后(n_example,n_out)矩阵的每一行都加b  
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # 通过符号变量来计算预测概率最大的类
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # 构成模型参数
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

        # 代价函数NLL  
        # 因为我们是MSGD，每次训练一个batch，一个batch有n_example个样本，则y大小是(n_example,),  
        # y.shape[0]得出行数即样本数，将T.log(self.p_y_given_x)简记为LP，  
        # 则LP[T.arange(y.shape[0]),y]得到[LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,LP[n-1,y[n-1]]]  
        # 最后求均值mean，也就是说，minibatch的SGD，是计算出batch里所有样本的NLL的平均值，作为它的cost  
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y): # batch的误差率  
        # 检测y与y_pred是否有相同的维度
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.y_pred.type)
            )
        # 检测y是否是正确的数据类型
        if y.dtype.startswith('int'): 
            # 再检查是不是int类型，是的话计算T.neq(self.y_pred, y)的均值，作为误差率  
            # 举个例子，假如self.y_pred=[3,2,3,2,3,2],而实际上y=[3,4,3,4,3,4]  
            # 则T.neq(self.y_pred, y)=[0,1,0,1,0,1],1表示不等，0表示相等  
            # 故T.mean(T.neq(self.y_pred, y))=T.mean([0,1,0,1,0,1])=0.5，即错误率50%            
            return T.mean(T.neq(self.y_pred, y)) # T.neq操作符返回了一个0和1的向量，这里1表示一个预测错误
        else:
            raise NotImplementedError()
    def save_net(self, path):  
        import cPickle  
        write_file = open(path, 'wb')   
        cPickle.dump(self.W.get_value(borrow=True), write_file, -1)  
        cPickle.dump(self.b.get_value(borrow=True), write_file, -1)  
        write_file.close()

##############
# 加载数据集 #
##############
def load_data(dataset):
    # 如果本地不存在就从网络下载数据集
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # 检测数据集是否在数据目录中.
        new_path = os.path.join(
            os.path.split(__file__)[0], "..", "data", dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = ( 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz' )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # 加载数据集主方法
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    # 函数闭包：训练集, 验证集, 测试集格式: tuple(input, target)
    # 输入时一个二维的numpy.ndarray (是一个矩阵)
    # 其中每一行都是一个样本. 目标是个一维的numpy.ndarray (vector))与输入行数有同样的长度 
    # 它应该给出一个与输入数据索引相同的样本目标。
    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        # 当存储数据到GPU时，数据必须为float格式，因此我们将标签存储为"floatX"
        # 但在计算期间，我们需要将其作为int，(因为是索引，如果他们是浮点数
        # 它是无意义的)，返回之后，我们将其转换为int。这是回避该问题的一个技巧
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
# 主执行函数
# learning_rate=0.13, ：梯度下降法的权重更新率
# n_epochs=1000, 最大迭代次数
# dataset='mnist.pkl.gz', 数据集
# batch_size=600 批大小
def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz',batch_size=600):

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # 计算有多少个minibatch，因为我们的优化算法是MSGD，是一个batch一个batch来计算cost的
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ############
    # 构建模型 #
    ############
    print '... building the model'
    
    # 设置变量，index表示minibatch的下标，x表示训练样本，y是对应的label  
    index = T.lscalar()  # 索引变量    
    x = T.matrix('x')   # 数据, 呈现为光栅图像
    y = T.ivector('y')  # 标签, 呈现为[INT]标签的1维向量

    # 实例化logisitic分类器，每个MNIST 图的尺寸为 28*28 ，x用作input初始化。 
    classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)
    # 计算代价，用y来初始化，而其实还有一个隐含的参数Input
    cost = classifier.negative_log_likelihood(y) # 负对数似然的代价
    # 定义测试模型：函数指针
    test_model = theano.function(
        inputs=[index], outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # 定义验证模型：函数指针
    validate_model = theano.function(
        inputs=[index], outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # 计算theta = (W,b)的梯度
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # learning_rate：梯度下降法的权重更新率
    updates = [(classifier.W, classifier.W - learning_rate * g_W),(classifier.b, classifier.b - learning_rate * g_b)]

    # 函数指针定义训练模型
    train_model = theano.function(
        inputs=[index], outputs=cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    #   训练模型  #
    ###############
    print '... training the model'
    patience = 5000  # 最大迭代次数
    patience_increase = 2   # 步长
    # 提高的阈值，在验证误差减小到之前的0.995倍时，会更新best_validation_loss     
    improvement_threshold = 0.995  # 相当大的改善被认为是显著
    # 这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。  
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = numpy.inf # 最好的验证集上的loss，越小即越好。初始化为无穷大  
    test_score = 0.
    start_time = time.clock() # 开始时间

    done_looping = False
    epoch = 0
    # 下面就是训练过程了，while循环控制的时步数epoch，一个epoch会遍历所有的batch，即所有的图片。 
    # 当达到最大步数n_epoch时，或者patience<iter时，结束训练  
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        # for循环是遍历一个个batch，一次一个batch地训练。          
        # 循环体里会用train_model(minibatch_index)去训练模型，
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)  # 得到每批的平均代价       
            iter = (epoch - 1) * n_train_batches + minibatch_index # 累加训练过的batch数iter。
            # 当iter是validation_frequency倍数时则会在验证集上测试，
            if (iter + 1) % validation_frequency == 0:                
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)] # 计算验证集的损失概率
                this_validation_loss = numpy.mean(validation_losses)
                # 输出
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,  minibatch_index + 1, n_train_batches,  this_validation_loss * 100.
                    )
                )
                # 如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，  
                # 则更新best_validation_loss和best_iter，同时在testset上测试。  
                # 如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch, minibatch_index + 1, n_train_batches, test_score * 100.
                        )
                    )
            if patience <= iter: # 达到最大迭代次数，退出
                done_looping = True
                break
    classifier.save_net("convnet.data") # 保存我们训练后神经网络的结果参数。
    # while循环结束
    end_time = time.clock() # 结束时间
    # 输出
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % ( epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))





if __name__ == '__main__':
    sgd_optimization_mnist()
