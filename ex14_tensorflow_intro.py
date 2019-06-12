import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# 1. simple nodes
node1 = tf.constant(5, name="node1")
node2 = tf.constant(11, name="node2")
h = tf.Variable(node1 + node2, name="h")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(h))

# 2. using X and Y input data as placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(7, name="W")
b = tf.Variable(8, name="b")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(W+b, feed_dict={X: 4, Y:2}))

# 3. difference between variable and placeholder
#print("A variable holds a value to be used in a process, and which will change throughout the process.",
#      "A placeholder indicates the place in memory where data will be fed into at the start of the process.")

#print("A variable requires an initial value.",
#      "A placeholder does not require an initial value.")

#print("A variable has a predetermined shape.",
#      "A placeholder can later take data of varying shape.")

# 4. Using loss
X = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
W = tf.Variable(7, name="W")
b = tf.Variable(8, name="b")
linearModel = W*X + b
squared_diff = tf.square(y - linearModel)
loss = tf.reduce_sum(squared_diff, 0)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#print(sess.run(loss, feed_dict={X: [1,2,3,4], y: [0, -1, -2, -3]}))

# 5. Using assign
assign_W = W.assign(-1)
assing_b = b.assign(1)
sess.run(assign_W)	
sess.run(assing_b)	
#print(sess.run(loss, feed_dict={X: [1,2,3,4], y: [0, -1, -2, -3]}))

# 6. Using gradient descent
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable(7., name="W")
b = tf.Variable(8., name="b")
linearModel = W*X + b
squared_diff = tf.square(y - linearModel)
loss = tf.reduce_sum(squared_diff, 0)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss)
loss_vect = []
x1 = [1.,2.,3.,4.]
y1 = [0, -1, -2, -3]
accepted_loss = .1

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(W)	
    session.run(b)	
    
    print("Starting values:", session.run(linearModel,feed_dict={X: x1, y: y1}))
    for step in range(1000):  
        _,current_loss = session.run([optimizer, loss],feed_dict={X: x1, y: y1})
        loss_vect.append(current_loss)
        if loss_vect[step] < accepted_loss:
            found_good_match = step+1
            break
        
    out = session.run(linearModel,feed_dict={X: x1, y: y1})
    result = np.round(out)
    print("Final Weights achived at iteration",found_good_match,": ",result)
    
plt.plot(loss_vect)
plt.show()

