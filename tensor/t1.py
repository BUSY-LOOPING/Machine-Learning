import numpy as np
import tensorflow as tf

string = tf.Variable("string", tf.string)
number = tf.Variable(1234,tf.int16)
floating = tf.Variable(3.14, tf.float64)
rank1 = tf.Variable(["rank1"], tf.string)
rank2 = tf.Variable([["rank2", "tensor"], ["second", "rank"]], tf.string)
tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1,[2, 3, 1])
tensor3 = tf.reshape(tensor1,[3, -1])
print(tensor1)
print(tensor2)
print(tensor3)
t = tf.zeros([5, 5])
print('t', t)