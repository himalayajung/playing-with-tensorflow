import tensorflow as tf

x = tf.placeholder("float") # create a symbolic variable
y = tf.placeholder("float")

z = tf.add(x, y) # create a symbolic expression

sess = tf.Session() 

print(sess.run(z, feed_dict={x: 3, y: 2})) # eval expressions with parameters x and y
#print(sess.run(z, feed_dict={p: 3, q: 0})) # NameError: name 'p' is not defined

print(sess.run(tf.mul(x,y), feed_dict={x: 3, y: 2})) # multiply

print(sess.run(tf.mul(x,y), {x: 3, y: 2})) # multiply
