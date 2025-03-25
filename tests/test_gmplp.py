import tensorflow as tf
from kdl_pre_built.layers import gmplp


def test_polynomial_dense():
    x = tf.random.normal(shape=[2, 16])
    output = gmplp.PolynomialDense(
        units=8,
        activation="relu",
        extra_args={}
    )(x)
    assert output.shape == (2, 8)
    

def test_gmplp_block():
    x = tf.random.normal(shape=[2, 16, 16])
    output = gmplp.GMPLPBlock(
        units=8,
        activation="relu",
        drop_rate=0.2,
        extra_args={}
    )(x)
    assert output.shape == (2, 16, 8)