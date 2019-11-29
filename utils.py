import tensorflow as tf
import numpy as np


def lognormdens(x, mu, logsigma):
    numerator = tf.reduce_sum(-.5 * (x - mu)**2 / tf.exp(logsigma * 2), 1)
    denominator = tf.reduce_sum(.5 * tf.log(2 * np.pi) + logsigma, 1)
    return numerator - denominator



def weight_variable(shape, stddev=1.0, name='weight'):
    initial = tf.random.normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, value=0.0, name='bias'):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)

