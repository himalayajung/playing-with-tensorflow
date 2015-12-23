import tensorflow as tf
import numpy as np

slope = 5
xx = np.linspace(-1, 1, 20)
yy = slope*xx + np.random.randn(*xx.shape)*0.01

# Create place holders (symbolic variables)
# Inserts a placeholder for a tensor that will be always fed.
# **Important**: This tensor will produce an error if evaluated. Its value must
# be fed using the `feed_dict` optional argument to `Session.run()`
# Returns a `Tensor` that may be used as a handle for feeding a value, but not
# evaluated directly.

X = tf.placeholder("float")   # create symbolic variables
Y = tf.placeholder("float")


def model(X, w):
    return tf.mul(X, w)  # defining the linear model as X*w


# Create a variable.
# w = tf.Variable(<initial-value>, name=<optional-name>)
# A variable maintains state in the graph across calls to `run()`. You add a
# variable to the graph by constructing an instance of the class `Variable`.

w = tf.Variable(0.0, name="weights")  # create a shared variable (like theano.shared) for the weight matrix
yf = model(X, w) # fitted values

cost = tf.pow(Y-yf, 2)  # cost function = sum of squares

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
sess = tf.Session()
init = tf.initialize_all_variables()  # initialize W
sess.run(init)

for i in range(50):
    for (x, y) in zip(xx, yy):
        sess.run(train_op, feed_dict={X: x, Y: y})

print(sess.run(w))  
