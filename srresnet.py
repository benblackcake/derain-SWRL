
import tensorflow as tf
from utils import tf_idwt, tf_dwt, tf_batch_ISwt

class Srresnet:
    """Srresnet Model"""

    def __init__(self, training, content_loss='mse', learning_rate=1e-4, num_blocks=16):
        self.learning_rate = learning_rate
        self.num_blocks = num_blocks
        self.training = training

        if content_loss not in ['mse', 'L1','edge_loss_mse','edge_loss_L1']:
            print('Invalid content loss function. Must be \'mse\', or \'L1_loss\'.')
            exit()
        self.content_loss = content_loss

    def ResidualBlock(self, x, kernel_size, filter_size):
        """Residual block a la ResNet"""
        # with tf.variable_scope('sr_edge_net') as scope:       
        weights = {
            'w1': tf.Variable(tf.random_normal([kernel_size, kernel_size,filter_size, filter_size], stddev=1e-3), name='w1_redidual'),
            'w2': tf.Variable(tf.random_normal([kernel_size, kernel_size,filter_size, filter_size], stddev=1e-3), name='w2_residual'),

        }

        skip = x
        x = tf.nn.conv2d(x, weights['w1'], strides=[1,1,1,1], padding='SAME')
        # x = tf.nn.atrous_conv2d(x,filters = weights['w1'], rate=4, padding='SAME')
        # x = tf.layers.batch_normalization(x, training=self.training)
        # x = tf.nn.relu(x)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        x = tf.nn.conv2d(x, weights['w2'], strides=[1,1,1,1], padding='SAME')
        # x = tf.nn.atrous_conv2d(x,filters = weights['w2'], rate=4, padding='SAME')
        # x = tf.nn.relu(x)
        x = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x)
        # x = tf.layers.batch_normalization(x, training=self.training)

        x = x + skip
        return x


    def RDBParams(self):
        weightsR = {}
        biasesR = {}
        D = self.num_blocks
        C = 8
        G = 64
        # C = 8
        # G = 32
        # G0 = self.G0
        ks = 3

        for i in range(1, D+1):
            for j in range(1, C+1):
                weightsR.update({'w_R_%d_%d' % (i, j): tf.Variable(tf.random_normal([ks, ks, G * j, G], stddev=0.01), name='w_R_%d_%d' % (i, j))}) 
                biasesR.update({'b_R_%d_%d' % (i, j): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, j)))})
            weightsR.update({'w_R_%d_%d' % (i, C+1): tf.Variable(tf.random_normal([1, 1, G * (C+1), G], stddev=0.01), name='w_R_%d_%d' % (i, C+1))})
            biasesR.update({'b_R_%d_%d' % (i, C+1): tf.Variable(tf.zeros([G], name='b_R_%d_%d' % (i, C+1)))})

        return weightsR, biasesR



    def forward_branch_bine(self, input_LL, input_edge):
        rdb_concat = list()
        rdb_in = input_edge
        x_LL = input_LL

        D = self.num_blocks
        C = 8
        G = 64
        # G0 = self.G0
        ks = 3
        for i in range(1, D+1):
            x_LL = self.ResidualBlock(x_LL, 3, 64)
            x_edge = rdb_in

            for j in range(1, C+1):
                tmp = tf.nn.conv2d(x_edge, self._weightsR['w_R_%d_%d' %(i, j)], strides=[1,1,1,1], padding='SAME') + self._biasesR['b_R_%d_%d' % (i, j)]
                tmp = tf.nn.relu(tmp)
                x_edge = tf.concat([x_edge, tmp], axis=3)

            x_edge = tf.nn.conv2d(x_edge, self._weightsR['w_R_%d_%d' % (i, C+1)], strides=[1,1,1,1], padding='SAME') +  self._biasesR['b_R_%d_%d' % (i, C+1)]

            rdb_in = tf.add(x_edge, rdb_in)
            rdb_in = tf.add(rdb_in, x_LL)

            rdb_concat.append(rdb_in)

        return tf.concat(rdb_concat, axis=3), x_LL


    def forward(self,x_LL, x_edge):
        with tf.variable_scope('forward_branch') as scope:
            weights = {
                'w_resnet_in': tf.Variable(tf.random_normal([9, 9, 3, 64], stddev=1e-2), name='w_resnet_in'),
                'w_resnet_1': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=1e-2), name='w_resnet_1'),
                'w_resnet_out': tf.Variable(tf.random_normal([9, 9, 64, 3], stddev=1e-2), name='w_resnet_out'),
                'w_RDB_in': tf.Variable(tf.random_normal([3, 3, 9, 64], stddev=1e-2), name='w_RDB_in'),
                'w_RDB_1': tf.Variable(tf.random_normal([3, 3, self.num_blocks*64, 64], stddev=1e-2), name='w_RDB_1'),
                'w_RDB_out': tf.Variable(tf.random_normal([3, 3, 64, 9], stddev=1e-2), name='w_RDB_out'),

            }
            biases = {
                'b1': tf.Variable(tf.zeros([64], name='b1')),
                'b2': tf.Variable(tf.zeros([64], name='b2')),
                'b3': tf.Variable(tf.zeros([9], name='b3')),
                }

            self._weightsR, self._biasesR = self.RDBParams()

            x_edge = tf.nn.conv2d(x_edge, weights['w_RDB_in'], strides=[1,1,1,1], padding='SAME') + biases['b1']
            x_edge_skip = x_edge

            x_LL = tf.nn.conv2d(x_LL, weights['w_resnet_in'], strides=[1,1,1,1], padding='SAME')
            x_LL = tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x_LL)
            x_LL_skip = x_LL

            x_edge, x_LL = self.forward_branch_bine(x_edge, x_LL)
            print('-----------=====debug',x_edge)

            x_edge = tf.nn.conv2d(x_edge, weights['w_RDB_1'], strides=[1,1,1,1], padding='SAME') + biases['b2']
            print('-----------=====debug',x_edge)

            x_edge =  tf.contrib.keras.layers.PReLU(shared_axes=[1, 2])(x_edge)
            x_edge = tf.add(x_edge_skip, x_edge)

            x_LL = tf.nn.conv2d(x_LL, weights['w_resnet_1'], strides=[1,1,1,1], padding='SAME', name='layer_1')
            # x_LL = tf.layers.batch_normalization(x_LL, training=self.training)
            x_LL = tf.add(x_LL_skip, x_LL)


            # for i in range(self.num_upsamples):
            #     x_edge = self.Upsample2xBlock(x_edge, kernel_size=3, in_channel=64, filter_size=256)
            #     x_LL = self.Upsample2xBlock(x_LL, kernel_size=3, in_channel=64, filter_size=256)

            x_edge = tf.nn.conv2d(x_edge, weights['w_RDB_out'], strides=[1,1,1,1], padding='SAME') + biases['b3']
            # x_edge = x_edge * 2.0

            x_LL = tf.nn.conv2d(x_LL, weights['w_resnet_out'], strides=[1,1,1,1], padding='SAME', name='y_predict')

            tf_swt_debug_RA = tf.expand_dims(x_LL[:,:,:,0], axis=-1)
            tf_swt_debug_GA = tf.expand_dims(x_LL[:,:,:,1], axis=-1)
            tf_swt_debug_BA = tf.expand_dims(x_LL[:,:,:,2], axis=-1)

            y_RA_pred = tf.concat([tf_swt_debug_RA,x_edge[:,:,:,0:3]], axis=-1)
            y_GA_pred = tf.concat([tf_swt_debug_GA,x_edge[:,:,:,3:6]], axis=-1)
            y_BA_pred = tf.concat([tf_swt_debug_BA,x_edge[:,:,:,6:9]], axis=-1)

            y_pred = tf.concat([y_RA_pred, y_GA_pred, y_BA_pred], axis=-1)

            y_pred = tf_batch_ISwt(y_pred)

            return x_LL, x_edge, y_pred
                
    def _content_loss(self, y_A, y_A_pred, y_BCD, y_BCD_pred, hr, sr_pred):


        """MSE, VGG22, or VGG54"""
        if self.content_loss == 'mse':
            return tf.reduce_mean(tf.square(hr - sr_pred))

        if self.content_loss == 'L1':
            return tf.reduce_mean(tf.abs(hr - sr_pred))

        if self.content_loss == 'edge_loss_mse':
            lamd = 0.05
            # y_sobeled = tf.image.sobel_edges(y)
            # y_pred_sobeled = tf.image.sobel_edges(y_pred)
            return tf.reduce_mean(tf.square(y_A - y_A_pred))\
             + (lamd*tf.reduce_mean(tf.square(y_BCD - y_BCD_pred)))

        if self.content_loss == 'edge_loss_L1':

            lamd_LL = 5e-1
            lamd_edge = 2e-1
            alpha = 1.0
            # y_sobeled = tf.image.sobel_edges(y)
            # y_pred_sobeled = tf.image.sobel_edges(y_pred)
            l_pred = tf.cast(hr - sr_pred, tf.float32)
            l_pred = tf.concat([l_pred, l_pred, l_pred], axis=-1)
            print(l_pred)
            return tf.cast(tf.reduce_mean(tf.square(hr - sr_pred)), tf.float32)\
             + lamd_LL * (tf.reduce_mean(tf.square(y_A - y_A_pred)))\
             + lamd_edge *tf.reduce_mean(tf.abs(alpha * y_BCD-(y_BCD_pred)))
             # + 1.0*tf.reduce_mean(y_BCD*tf.cast((hr - sr_pred), tf.float32))
        # if self.content_loss == 'edge_loss_L1':
        #     lamd = 2.0
        #     # y_sobeled = tf.image.sobel_edges(y)
        #     # y_pred_sobeled = tf.image.sobel_edges(y_pred)
        #     return tf.reduce_mean(tf.square(y_A - y_A_pred)) + (lamd*tf.reduce_mean(tf.abs(y_BCD - y_BCD_pred)))

    def loss_function(self, y_A, y_A_pred, y_BCD, y_BCD_pred, hr, sr_pred):

        # Content loss only
        return self._content_loss(y_A, y_A_pred, y_BCD, y_BCD_pred, hr, sr_pred)

    def optimize(self, loss):
        # tf.control_dependencies([discrim_train
        # update_ops needs to be here for batch normalization to work
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='forward_branch')
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='forward_branch'))



