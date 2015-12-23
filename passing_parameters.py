import tensorflow as tf

x = tf.placeholder("float") # Create a symbolic variable 'a'
y = tf.placeholder("float") # Create a symbolic variable 'b'

z = tf.add(x, y)

sess = tf.Session() 

print(sess.run(z, feed_dict={x: 1, y: 2})) # eval expressions with parameters for a and b
print(sess.run(z, feed_dict={p: 3, q: 3})) # NameError: name 'p' is not defined