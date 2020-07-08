import math
import random
import tensorflow as tf
import keras.backend as K
from keras.layers import GaussianNoise, Dense, Activation
from scSemiCluster_preprocess import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os
import time


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def label2matrix(label):
    unique_label, label = np.unique(label, return_inverse=True)
    one_hot_label = np.zeros((len(label), len(unique_label)))
    one_hot_label[np.arange(len(label)), label] = 1
    return one_hot_label


def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)


def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)


def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)


def cal_centers(repre, label):
    class_number = len(np.unique(label))
    centers = np.zeros((class_number, repre.shape[1]))
    for i in range(class_number):
        centers[i] = np.mean(repre[label == i], axis=0)
    return centers


def cross_entropy_dec(hidden, cluster, alpha = 1.0):
    dist = K.sum(K.square(K.expand_dims(hidden, axis=1) - cluster), axis=2)
    q = 1.0 / (1.0 + dist / alpha) ** ((alpha + 1.0) / 2.0)
    q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    p = q ** 2 / tf.reduce_sum(q, axis=0)
    p = p / tf.reduce_sum(p, axis=1, keepdims=True)
    crossentropy = -p * tf.log(tf.clip_by_value(q, 1e-10, 1.0)) - (1 - p) * tf.log(tf.clip_by_value(1 - q, 1e-10, 1.0))
    return dist, crossentropy


def dec(hidden, cluster, alpha = 1.0):
    dist = K.sum(K.square(K.expand_dims(hidden, axis=1) - cluster), axis=2)
    q = 1.0 / (1.0 + dist / alpha) ** ((alpha + 1.0) / 2.0)
    q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    p = q ** 2 / tf.reduce_sum(q, axis=0)
    p = p / tf.reduce_sum(p, axis=1, keepdims=True)
    kl = p * tf.log(tf.clip_by_value(p, 1e-10, 1.0)) - p * tf.log(tf.clip_by_value(q, 1e-10, 1.0))
    return dist, kl


def NB(theta, y_true, y_pred, mask = False, debug = False, mean = False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = tf.minimum(theta, 1e6)
    t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
    if debug:
        assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                      tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                      tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
        with tf.control_dependencies(assert_ops):
            final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final


def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * tf.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result


class scSemiCluster(object):
    def __init__(self, dataname, dims, cluster_num, alpha, beta, gamma, learning_rate,
                 noise_sd=1.5, init='glorot_uniform', act='relu', mode="cdec", distrib = "ZINB", constraint = True):
        self.dataname = dataname
        self.dims = dims
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act
        self.mode = mode
        self.distrib = distrib
        self.constraint = constraint

        self.n_stacks = len(self.dims) - 1
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.cluster_num))
        self.label_vec = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.mask_vec = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.x_count = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.sf_layer = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[self.cluster_num, self.dims[-1]],
                                        dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

        self.label_mat = tf.reshape(self.label_vec, [-1, 1]) - tf.reshape(self.label_vec, [1, -1])
        self.label_mat = tf.cast(tf.equal(self.label_mat, 0.), tf.float32)
        self.mask_mat = tf.matmul(tf.reshape(self.mask_vec, [-1, 1]), tf.reshape(self.mask_vec, [1, -1]))

        self.h = self.x
        self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)

        for i in range(self.n_stacks - 1):
            self.h = Dense(units=self.dims[i + 1], kernel_initializer=self.init, name='encoder_%d' % i)(self.h)
            self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
            self.h = Activation(self.act)(self.h)

        self.latent = Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')(self.h)

        self.discriminate = Dense(units=self.cluster_num, activation=tf.nn.softmax, kernel_initializer=self.init,
                                  name='classification_layer')(self.latent)
        self.softmax_mat = -self.y * tf.log(tf.clip_by_value(self.discriminate, 1e-10, 1.0))
        self.softmax_loss = tf.reduce_sum(self.softmax_mat)   #self.softmax_loss = tf.reduce_mean(tf.reduce_sum(self.softmax_mat, axis=1))

        self.normalize_discriminate = tf.nn.l2_normalize(self.discriminate, axis=1)
        self.similarity = tf.matmul(self.normalize_discriminate, tf.transpose(self.normalize_discriminate))
        self.cross_entropy = self.mask_mat * (-self.label_mat * tf.log(tf.clip_by_value(self.similarity, 1e-10, 1.0)) -
                                              (1 - self.label_mat) * tf.log(tf.clip_by_value(1 - self.similarity, 1e-10, 1.0)))
        self.cross_entropy = tf.reduce_sum(self.cross_entropy)   #self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis=1))

        if self.mode == "cdec":
            self.latent_dist1, self.latent_dist2 = cross_entropy_dec(self.latent, self.clusters)
        elif self.mode == "dec":
            self.latent_dist1, self.latent_dist2 = dec(self.latent, self.clusters)

        self.h = self.latent
        for i in range(self.n_stacks - 1, 0, -1):
            self.h = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init, name='decoder_%d' % i)(self.h)

        if self.distrib == "ZINB":
            self.pi = Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
            self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.ZINB_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)
            self.pre_loss = self.ZINB_loss
        elif self.distrib == "NB":
            self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.NB_loss = NB(self.disp, self.x_count, self.output, mask=False, debug=False, mean=True)
            self.pre_loss = self.NB_loss

        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))
        if self.constraint:
            self.mid_loss1 = self.pre_loss + self.alpha * self.softmax_loss
            self.mid_loss2 = self.mid_loss1 + self.beta * self.cross_entropy
            self.total_loss = self.mid_loss2 + self.gamma * self.kmeans_loss
        else:
            self.mid_loss1 = self.pre_loss + self.alpha * self.softmax_loss
            self.total_loss = self.mid_loss1 + self.gamma * self.kmeans_loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.pretrain_op = self.optimizer.minimize(self.pre_loss)
        if self.constraint:
            self.midtrain_op1 = self.optimizer.minimize(self.mid_loss1)
            self.midtrain_op2 = self.optimizer.minimize(self.mid_loss2)
        else:
            self.midtrain_op1 = self.optimizer.minimize(self.mid_loss1)
        self.train_op = self.optimizer.minimize(self.total_loss)


    def train(self, X, count_X, size_factor, cellname, batch_label, batch_size, random_seed, gpu_option):
        t1 = time.time()
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        pretrain_epochs = 100
        if self.constraint:
            midtrain_epochs1 = 400
            midtrain_epochs2 = 500
        else:
            midtrain_epochs1 = 900
        funetrain_epochs = 2000
        tol = 0.001

        cell_type, Y = np.unique(cellname, return_inverse=True)
        Y_train = Y[batch_label == 0]
        cellname_train = cellname[batch_label == 0]

        Y_test = Y[batch_label != 0]
        cellname_test = cellname[batch_label != 0]

        label_vec = Y.astype(np.float32)
        mask_vec = batch_label - 1 + 1
        mask_vec[mask_vec != 0] = 1
        mask_vec = (1 - mask_vec).astype(np.float32)

        cluster_num_train = len(np.unique(Y_train))
        cluster_num_test = len(np.unique(Y_test))
        n_clusters = len(np.unique(Y))
        n_batches = len(np.unique(batch_label))
        cluster_num_overlap = cluster_num_train + cluster_num_test - n_clusters

        onehot_Y = label2matrix(Y)
        onehot_Y[batch_label != 0] = np.zeros((len(Y_test), n_clusters))
        onehot_Y = onehot_Y.astype(np.float32)

        if X.shape[0] < batch_size:
            batch_size = X.shape[0]

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option

        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        sess = tf.Session(config=config_)
        sess.run(init)

        latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        iteration_per_epoch = math.ceil(float(len(X)) / float(batch_size))
        for i in range(pretrain_epochs):
            for j in range(iteration_per_epoch):
                batch_idx = random.sample(range(X.shape[0]), batch_size)
                _, latent, likeli = sess.run(
                    [self.pretrain_op, self.latent, self.pre_loss],
                    feed_dict={
                        self.sf_layer: size_factor[batch_idx],
                        self.x: X[batch_idx],
                        self.y: onehot_Y[batch_idx],
                        self.x_count: count_X[batch_idx],
                        self.label_vec: label_vec[batch_idx],
                        self.mask_vec: mask_vec[batch_idx]})
                latent_repre[batch_idx] = latent

        for i in range(midtrain_epochs1):
            for j in range(iteration_per_epoch):
                batch_idx = random.sample(range(X.shape[0]), batch_size)
                _, latent, likeli, softmax, entropy = sess.run(
                    [self.midtrain_op1, self.latent, self.pre_loss, self.softmax_loss, self.cross_entropy],
                    feed_dict={
                        self.sf_layer: size_factor[batch_idx],
                        self.x: X[batch_idx],
                        self.y: onehot_Y[batch_idx],
                        self.x_count: count_X[batch_idx],
                        self.label_vec: label_vec[batch_idx],
                        self.mask_vec: mask_vec[batch_idx]})
                latent_repre[batch_idx] = latent

        if self.constraint:
            for i in range(midtrain_epochs2):
                for j in range(iteration_per_epoch):
                    batch_idx = random.sample(range(X.shape[0]), batch_size)
                    _, latent, likeli, softmax, entropy = sess.run(
                        [self.midtrain_op2, self.latent, self.pre_loss, self.softmax_loss, self.cross_entropy],
                        feed_dict={
                            self.sf_layer: size_factor[batch_idx],
                            self.x: X[batch_idx],
                            self.y: onehot_Y[batch_idx],
                            self.x_count: count_X[batch_idx],
                            self.label_vec: label_vec[batch_idx],
                            self.mask_vec: mask_vec[batch_idx]})
                    latent_repre[batch_idx] = latent

        if self.gamma == 0.:
            dist, discrimin, kmeans_loss = sess.run(
                [self.latent_dist1, self.discriminate, self.kmeans_loss],
                feed_dict={
                    self.sf_layer: size_factor,
                    self.x: X,
                    self.y: onehot_Y,
                    self.x_count: count_X,
                    self.label_vec: label_vec,
                    self.mask_vec: mask_vec})
            Y_pred = np.argmax(discrimin, axis=1)
            Y_pred_train = Y_pred[batch_label == 0]
            Y_pred_test = Y_pred[batch_label != 0]
        else:
            latent_repre = np.nan_to_num(latent_repre)
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
            kmeans_pred = kmeans.fit_predict(latent_repre)
            kmeans_pred_test = kmeans_pred[batch_label == 1]
            last_pred_test = np.copy(kmeans_pred_test)
            latent_repre_train = latent_repre[batch_label == 0]
            cluster_centers = cal_centers(latent_repre_train, Y_train)
            sess.run(tf.assign(self.clusters, cluster_centers))
            for i in range(funetrain_epochs):
                if (i + 1) % 10 != 0:
                    for j in range(iteration_per_epoch):
                        batch_idx = random.sample(range(X.shape[0]), batch_size)
                        _, Kmeans_loss, latent = sess.run(
                            [self.train_op, self.kmeans_loss, self.latent],
                            feed_dict={
                                self.sf_layer: size_factor[batch_idx],
                                self.x: X[batch_idx],
                                self.y: onehot_Y[batch_idx],
                                self.x_count: count_X[batch_idx],
                                self.label_vec: label_vec[batch_idx],
                                self.mask_vec: mask_vec[batch_idx]})
                        latent_repre[batch_idx] = latent
                else:
                    dist, discrimin, kmeans_loss = sess.run(
                        [self.latent_dist1, self.discriminate, self.kmeans_loss],
                        feed_dict={
                            self.sf_layer: size_factor,
                            self.x: X,
                            self.y: onehot_Y,
                            self.x_count: count_X,
                            self.label_vec: label_vec,
                            self.mask_vec: mask_vec})
                    Y_pred = np.argmin(dist, axis=1)
                    Y_pred_train = Y_pred[batch_label == 0]
                    Y_pred_test = Y_pred[batch_label != 0]
                    if np.sum(Y_pred_test != last_pred_test) / len(last_pred_test) < tol:
                        break
                    else:
                        last_pred_test = Y_pred_test

        sess.close()
        test_ARI = np.around(adjusted_rand_score(Y_test, Y_pred_test), 4)
        pred_cluster_num = len(np.unique(np.array(Y_pred_test)))
        annotated_train_accuracy, annotated_test_accuracy, test_annotation_label = annotation(cellname_train, cellname_test, Y_pred_train, Y_pred_test)

        test_prediction_matrix = pd.DataFrame({"true label": Y_test, "true cell type": cellname_test, "cluster label": Y_pred_test,
                                               "annotation cell type": test_annotation_label})
        t2 = time.time()
        print("The total consuming time for the whole model training is {}".format(t2 - t1))
        return annotated_test_accuracy, test_ARI, test_prediction_matrix


if __name__ == "__main__":
    random_seed = 8888
    gpu_option = "0"
    dataname = "splatter_cluster_num_7_size_smaller_mode_imbalance_dropout_rate_0.5_data_5.h5"
    X, Y, batch_label = read_simu(dataname)
    Y = Y.astype(np.int)
    cellname = np.array(["group" + str(i) for i in Y])
    dims = [1000, 256, 64, 32]
    batch_size = 256
    highly_genes = 1000
    count_X = X
    if X.shape[1] == highly_genes:
        highly_genes = None
    print("begin the data proprocess")
    adata = sc.AnnData(X)
    adata.obs["celltype"] = cellname
    adata.obs["batch"] = batch_label
    adata = normalize(adata, highly_genes=highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    X = adata.X.astype(np.float32)
    cellname = np.array(adata.obs["celltype"])
    batch_label = np.array(adata.obs["batch"])

    if highly_genes != None:
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
        count_X = count_X[:, high_variable]
    else:
        select_genes = np.array(adata.var.index, dtype=np.int)
        select_cells = np.array(adata.obs.index, dtype=np.int)
        count_X = count_X[:, select_genes]
        count_X = count_X[select_cells]
    assert X.shape == count_X.shape
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)

    result = []
    cluster_num = len(np.unique(cellname))
    tf.reset_default_graph()
    scSemi = scSemiCluster(dataname, dims, cluster_num, 0.01, 0.1, 0.1, 1e-4, mode="cdec", distrib = "ZINB", constraint = True)
    annotated_target_accuracy, target_ARI, target_prediction_matrix = scSemi.train(X, count_X, size_factor, cellname, batch_label,
                                                                                   batch_size, random_seed, gpu_option)
    print("Under this setting, for target data, the clustering ARI is {}".format(target_ARI))
    print("Under this setting, for target data, the annotation accuracy is {}".format(annotated_target_accuracy))
    print("The target prediction information is in the target_prediction_matrix. It is a data frame, include four columns, "
          "they are true label, true cell type, cluster label, annotation cell type. You can save it in .csv file.")
    print(target_prediction_matrix)















