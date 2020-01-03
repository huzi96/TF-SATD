import numpy as np
import tensorflow as tf

def SATD(y_true, y_pred, scale, batch_size=1, norm='L1'):
  H_8x8 = np.array(
      [[1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [1., -1.,  1., -1.,  1., -1.,  1., -1.],
        [1.,  1., -1., -1.,  1.,  1., -1., -1.],
        [1., -1., -1.,  1.,  1., -1., -1.,  1.],
        [1.,  1.,  1.,  1., -1., -1., -1., -1.],
        [1., -1.,  1., -1., -1.,  1., -1.,  1.],
        [1.,  1., -1., -1., -1., -1.,  1.,  1.],
        [1., -1., -1.,  1., -1.,  1.,  1., -1.]],
      dtype=np.float32
  )
  H_target = np.zeros((1, 32,32), dtype=np.float32)
  H_target[0, 0:8,0:8] = H_8x8

  H_target[0, 0:8,8:16] = H_8x8
  H_target[0, 8:16,0:8] = H_8x8
  H_target[0, 8:16,8:16] = -H_8x8

  H_target[0, 16:32, 0:16] = H_target[0, 0:16, 0:16]
  H_target[0, 0:16, 16:32] = H_target[0, 0:16, 0:16]
  H_target[0, 16:32, 16:32] = -H_target[0, 0:16, 0:16]

  TH0 = tf.constant(H_target[:, 0:scale, 0:scale])

  TH1 = tf.tile(TH0, (batch_size, 1, 1))

  diff = tf.reshape(y_true - y_pred, (-1, scale, scale))

  if norm == 'L1':
    return tf.reduce_mean(tf.abs(tf.matmul(tf.matmul(TH1, diff), TH1)))
  elif norm == 'L2':
    return tf.reduce_mean(tf.square(tf.matmul(tf.matmul(TH1, diff), TH1)))
  elif norm == 'None':
    return tf.matmul(tf.matmul(TH1, diff), TH1)
  else:
    return None

if __name__ == "__main__":
    # Testing
    y_pred = np.ones((1, 8, 8), np.float32)
    y_true = np.zeros((1, 8, 8), np.float32)
    loss = SATD(y_true, y_pred, 8, norm='L1')
    with tf.Session() as sess:
      v_satd = sess.run(loss)
    
    print(f'Loss Value: {v_satd}')
