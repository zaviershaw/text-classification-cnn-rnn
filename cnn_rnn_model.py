#encoding:utf-8
import  tensorflow as tf

class TextConfig():

    embedding_size=100     #dimension of word embedding
    vocab_size=8000        #number of vocabulary
    pre_trianing = None   #use vector_char trained by word2vec

    seq_length=200         #max length of sentence
    num_classes=2          #number of labels

    num_filters=128        #number of convolution kernel
    filter_sizes=[2,3,4]   #size of convolution kernel
    num_layers= 1           #the number of layer
    hidden_dim = 128       #the number of hidden units
    attention_size = 100    #the size of attention layer


    keep_prob=0.5          #droppout
    lr= 1e-3                #learning rate
    lr_decay= 0.8          #learning rate decay
    decay_steps = 100      #decay iterations steps
    clip= 6.0              #gradient clipping threshold
    l2_reg_lambda=0.01     #l2 regularization lambda

    num_epochs=10          #epochs
    batch_size=64          #batch_size
    print_per_batch =50   #print result
    save_per_batch = 10    #save result to tensorboard

    train_filename='./data/email_train.txt'  #train data
    test_filename='./data/email_test.txt'    #test data
    val_filename='./data/email_val.txt'      #validation data
    vocab_filename='./data/vocab.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file

class TextCNN_RNN(object):

    def __init__(self,config):

        self.config=config

        self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_y')
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        #self.l2_loss = tf.constant(0.0)

        self.cnn_rnn()
    def cnn_rnn(self):
        # 模型结构：词嵌入-进行不同尺寸卷积池化-全连接 ---拼接-全连接
        #                -双向GRU-全连接
        
        # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                initializer=tf.constant_initializer(self.config.pre_trianing))
            self.embedding_inputs= tf.nn.embedding_lookup(self.embedding, self.input_x)
            #self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                #Defining regularization functions
                #regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                # CNN layer
                conv = tf.layers.conv1d(self.embedding_inputs, self.config.num_filters, filter_size,name='conv-%s' % filter_size)#kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
                pooled_outputs.append(gmp)  

            self.h_pool = tf.concat(pooled_outputs, 1)
            #self.outputs= tf.reshape(self.h_pool, [-1, num_filters_total])

        # Fully connected layer
        with tf.name_scope("cnn_output"):
            # 全连接层，后面接dropout以及relu激活
            cnn_out = tf.layers.dense(self.h_pool, self.config.hidden_dim, name='cnn_out')
            cnn_out = tf.contrib.layers.dropout(cnn_out, self.keep_prob)
            cnn_out = tf.nn.relu(cnn_out)

        def basic_rnn_cell(rnn_size):
            # return tf.contrib.rnn.GRUCell(rnn_size)
            return tf.contrib.rnn.LSTMCell(rnn_size,state_is_tuple=True)

        # Define Forward RNN Cell
        # 
        with tf.name_scope('fw_rnn'):
            fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            fw_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, output_keep_prob=self.keep_prob)

        # Define Backward RNN Cell
        with tf.name_scope('bw_rnn'):
            bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            bw_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, output_keep_prob=self.keep_prob)

        with tf.name_scope('bi_rnn'):
            # rnn_output, _ = tf.nn.dynamic_rnn(fw_rnn_cell, inputs=embedding_inputs, sequence_length=self.seq_len, dtype=tf.float32)
            rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, inputs=self.embedding_inputs,
                                                            sequence_length=self.sequence_lengths, dtype=tf.float32)
        if isinstance(rnn_output, tuple):
            rnn_output = tf.concat(rnn_output, 2)

        # Attention Layer
        with tf.name_scope('attention'):
            input_shape = rnn_output.shape  # (batch_size, sequence_length, hidden_size)
            sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
            hidden_size = input_shape[2].value  # hidden size of the RNN layer
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([self.config.attention_size], stddev=0.1), name='attention_u')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)
            # Transform to batch_size * sequence_size
            attention_z = tf.concat(z_list, axis=1)
            self.alpha = tf.nn.softmax(attention_z)
            attention_output = tf.reduce_sum(rnn_output * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)

        # Fully connected layer
        with tf.name_scope("rnn_output"):
            # 全连接层，后面接dropout以及relu激活
            rnn_out = tf.layers.dense(attention_output, self.config.hidden_dim, name='rnn_out')
            rnn_out = tf.contrib.layers.dropout(rnn_out, self.keep_prob)
            rnn_out = tf.nn.relu(rnn_out)

        with tf.name_scope("outputs"):
            con = tf.concat([cnn_out, rnn_out], axis=-1)
            final_out = tf.layers.dense(con, self.config.hidden_dim, name='con_out')


        with tf.name_scope("classifier"):
            self.logits = tf.layers.dense(final_out, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        #Calculate cross-entropy loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #self.l2_loss = tf.losses.get_regularization_loss()
            #self.loss = tf.reduce_mean(cross_entropy)+self.l2_loss
            self.loss = tf.reduce_mean(cross_entropy)


        #Create optimizer
        with tf.name_scope('optimizer'):
            #学习率衰减
            starter_learning_rate = self.config.lr
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,self.config.decay_steps, self.config.lr_decay, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)   #self.config.lr
            #compute_gradients()计算梯度
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            #clip_by_global_norm:修正梯度值
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            #apply_gradients()应用梯度
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        #准确率
        with tf.name_scope('accuracy'):
            correct_pred=tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


