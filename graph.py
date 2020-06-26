import tensorflow as tf
import numpy as np

xy = np.loadtxt('./board/631633 cmap rand.csv', delimiter=',', dtype=np.int32)
y_data = xy[0:,1]

a = tf.placeholder(tf.int32)

c_summary = tf.summary.scalar('value', a)
merged = tf.summary.merge_all()
print("Done")

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./board/crand', sess.graph)
    sess.run(tf.initialize_all_variables())
    for step, y in enumerate(y_data):
        sess.run(a, feed_dict={a:y})
        result = sess.run(merged, feed_dict={a:y})
        writer.add_summary(result, step)


#
# import tensorflow as tf
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# add = tf.add(X, Y)
# mul = tf.multiply(X, Y)
#
# # step 1: node 선택
# add_hist = tf.summary.scalar('add_scalar', add)
# mul_hist = tf.summary.scalar('mul_scalar', mul)
#
# # step 2: summary 통합. 두 개의 코드 모두 동작.
# merged = tf.summary.merge_all()
# # merged = tf.summary.merge([add_hist, mul_hist])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     # step 3: writer 생성
#     writer = tf.summary.FileWriter('./board/sample_1', sess.graph)
#
#     for step in range(100):
#         # step 4: 노드 추가
#         summary = sess.run(merged, feed_dict={X: step * 1.0, Y: 2.0})
#         writer.add_summary(summary, step)

