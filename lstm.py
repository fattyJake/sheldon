# -*- coding: utf-8 -*-
###############################################################################
# Module:      Neural Network Structure
# Description: Recurrent Neural Network class with embedding
# Authors:     Yage Wang
# Created:     6.28.2018
###############################################################################

import os
import tensorflow as tf
from datetime import datetime
import numpy as np


class BiLSTM(object):
    """
    A Bi-LSTM for EHR classification
    Uses an embedding layer, followed by a bi-lstm, lstm, fully-connected and sofrmax layer

    Parameters
    ----------
    max_sequence_length: int
        fixed padding latest number of time buckets

    hidden_size: int
        number of T_LSTM units

    num_classes: int
        the number of y classes

    embedding_matrix: 2-D numpy array (vocab_size, embedding_size)
        random initialzed embedding matrix

    learning_rate: float
        initial learning rate for Adam Optimizer

    decay_steps: int
        step frequency to decay the learning rate. e.g. if 5000, model will reduce learning rate by decay_rate every 5000 trained batches

    decay_rate: float
        percentage of learning rate decay rate

    dropout_keep_prob: float
        percentage of neurons to keep from dropout regularization each layer

    l2_reg_lambda: float, default 0
        L2 regularization lambda for fully-connected layer to prevent potential overfitting

    objective: str, default 'ce'
        the objective function (loss) model trains on; if 'ce', use cross-entropy, if 'auc', use AUROC as objective

    initializer: tf tensor initializer object, default tf.random_normal_initializer(stddev=0.1)
        initializer for fully connected layer weights

    sec_order: list of strings, default ['MED','ALG','IMU','LAB','ECT','PRL','PCD','HTR','VIT','TXT']
        list of section codes under fixed order which consistenly assigned to tensors thought whole pipeline

    Examples
    --------
    >>> from sheldon.tlstm import T_LSTM
    >>> rnn = T_LSTM(max_sequence_length=200, hidden_size=128,
            num_classes=2, embedding_dict=embedding_dict,
            w_embedding_dict=w_embedding_dict, learning_rate=0.05,
            decay_steps=5000, decay_rate=0.9,
            dropout_keep_prob=0.8, l2_reg_lambda=0.0,
            objective='ce')
    """

    def __init__(
        self,
        max_sequence_length,
        hidden_size,
        num_classes,
        variable_size,
        embedding_size,
        learning_rate,
        decay_steps,
        decay_rate,
        dropout_keep_prob,
        l2_reg_lambda=0.0,
        objective="ce",
        initializer=tf.random_normal_initializer(stddev=0.1),
    ):

        """init all hyperparameter here"""
        tf.reset_default_graph()

        # set hyperparamter
        self.num_classes = num_classes
        self.variable_size = variable_size
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.dropout_keep_prob = dropout_keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.initializer = initializer

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1))
        )
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        # add placeholder (X, quantity, time and label)
        self.input_x = tf.placeholder(
            tf.int32, [None, self.max_sequence_length], name="input_x"
        )  # X [instance_size, num_bucket]
        self.input_q = tf.placeholder(
            tf.float32, [None, self.max_sequence_length], name="input_q"
        )  # Q [instance_size, num_bucket]
        self.input_y = tf.placeholder(
            tf.int8, [None, self.num_classes], name="input_y"
        )  # y [instance_size, num_classes]

        """define all weights here"""
        with tf.name_scope("embedding"), tf.device(
            "/gpu:0"
        ):  # embedding matrix
            embedding_matrix = tf.random_normal(
                (self.variable_size, self.embedding_size), stddev=0.1
            )
            embedding_matrix = tf.concat(
                [embedding_matrix, tf.zeros((1, self.embedding_size))], axis=0
            )
            self.Embedding = tf.Variable(
                embedding_matrix,
                trainable=True,
                dtype=tf.float32,
                name="embedding",
            )

            self.W_projection = tf.get_variable(
                "W_projection",
                shape=[self.hidden_size * 2, self.num_classes],
                initializer=self.initializer,
            )  # [embedding_size,label_size]
            self.b_projection = tf.get_variable(
                "b_projection", shape=[self.num_classes]
            )  # [label_size]

        # main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax
        # 1.get emebedding of words in the sentence
        embedded_X = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        # concatenate quantity
        embedded_X = tf.concat(
            [embedded_X, tf.expand_dims(self.input_q, axis=2)], axis=2
        )
        self.input = tf.concat(
            values=embedded_X, axis=2
        )  # shape: [batch_size, max_sequence_length, concate_embedding_size]

        # 2. Bi-LSTM layer
        # define lstm cess:get lstm cell output
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(
            self.hidden_size
        )  # forward direction cell
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(
            self.hidden_size
        )  # backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_fw_cell, output_keep_prob=self.dropout_keep_prob
            )
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_bw_cell, output_keep_prob=self.dropout_keep_prob
            )
        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size]
        #                            output: A tuple (outputs, output_states)
        #                                    where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_fw_cell, lstm_bw_cell, self.input, dtype=tf.float32
        )  # [batch_size,sequence_length,hidden_size] #creates a dynamic bidirectional recurrent neural network
        output_rnn = tf.concat(
            outputs, axis=2
        )  # [batch_size,sequence_length,hidden_size*2]

        # 3. Second LSTM layer
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size * 2)
        if self.dropout_keep_prob is not None:
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_cell, output_keep_prob=self.dropout_keep_prob
            )
        _, final_state_c_h = tf.nn.dynamic_rnn(
            rnn_cell, output_rnn, dtype=tf.float32
        )
        final_state = final_state_c_h[1]

        # 4 .FC layer
        self.output_rnn_last = tf.layers.dense(
            final_state, self.hidden_size * 2, activation=tf.nn.tanh
        )

        # 5. logits(use linear layer)
        with tf.name_scope(
            "output"
        ):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            self.logits = tf.nn.xw_plus_b(
                self.output_rnn_last,
                self.W_projection,
                self.b_projection,
                name="scores",
            )
            self.probs = tf.sigmoid(self.logits, name="probs")

        assert objective in [
            "ce",
            "auc",
        ], 'AttributeError: objective only acccept "ce" or "auc", got {}'.format(
            str(objective)
        )
        if objective == "ce":
            self.loss_val = self._loss(self.l2_reg_lambda)
        if objective == "auc":
            self.loss_val = self._loss_roc_auc(self.l2_reg_lambda)
        self.train_op = self._train()
        self.predictions = tf.argmax(
            self.logits, axis=1, name="predictions"
        )  # shape:[None,]

        # performance
        with tf.name_scope("performance"):
            _, self.auc = tf.metrics.auc(
                self.input_y, self.probs, curve="ROC", name="auc"
            )
            correct_prediction = tf.equal(
                tf.cast(self.predictions, tf.int8), self.input_y
            )  # tf.argmax(self.logits, 1)-->[batch_size]
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name="accuracy"
            )  # shape=()

    def _train(self):
        """
        based on the loss, use Adam to update parameter
        """
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True,
        )
        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer="Adam",
        )
        return train_op

    def _loss(self, l2_reg_lambda):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.input_y, logits=self.logits
            )
            loss = tf.reduce_mean(losses)
            l2_losses = (
                tf.add_n(
                    [
                        tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if "bias" not in v.name
                    ]
                )
                * l2_reg_lambda
            )
            loss = loss + l2_losses
        return loss

    def _loss_roc_auc(self, l2_reg_lambda):
        """
        ROC AUC Score.
        Approximates the Area Under Curve score, using approximation based on
        the Wilcoxon-Mann-Whitney U statistic.
        Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
        Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
        Measures overall performance for a full range of threshold levels.
        """
        pos = tf.boolean_mask(self.logits, tf.cast(self.input_y, tf.bool))
        neg = tf.boolean_mask(self.logits, ~tf.cast(self.input_y, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma
        masked = tf.boolean_mask(difference, difference < 0.0)
        loss = tf.reduce_sum(tf.pow(-masked, p))
        l2_losses = (
            tf.add_n(
                [
                    tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if "bias" not in v.name
                ]
            )
            * l2_reg_lambda
        )
        loss = loss + l2_losses
        return loss


def train_rnn(
    model,
    x_train,
    q_train,
    y_train,
    dev_sample_percentage,
    num_epochs,
    batch_size,
    evaluate_every,
    model_path,
):
    """
    Training module for BiLSTM objectives
    
    Parameters
    ----------
    model: object of T_LSTM
        initialized Phased LSTM model

    x_train: 4-D numpy array, shape (num_exemplars, num_bucket, num_section, max_token_length)
        variable indices all buckets and sections

    q_train: 4-D numpy array, shape (num_exemplars, num_bucket, num_section, max_token_length)
        quantities or time deltas of corresponding x_train

    y_train: 2-D numpy array, shape (num_exemplars, num_classes)
        whole training ground truth
        
    dev_sample_percentage: float
        percentage of x_train seperated from training process and used for validation

    num_epochs: int
        number of epochs of training, one epoch means finishing training entire training set

    batch_size: int
        size of training batches, this won't affect training speed significantly; smaller batch leads to more regularization

    evaluate_every: int
        number of steps to perform a evaluation on development (validation) set and print out info

    model_path: str
        the path to store the model

    Examples
    --------
    >>> from sheldon.tlstm import train_rnn
    >>> train_rnn(model=rnn,
                x_train=X, q_train=Q,
                y_train=y, dev_sample_percentage=0.01,
                num_epochs=20, batch_size=64,
                evaluate_every=100, model_path='./plstm_model/')
    """

    # get number of input exemplars
    training_size = y_train.shape[0]

    # shuffle and partition the input into trainging set and developmenet set
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(y_train.shape[0]))
    x_train = x_train[shuffle_indices]
    q_train = q_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    dev_sample_index = -1 * int(dev_sample_percentage * float(training_size))
    x_train, x_dev = x_train[:dev_sample_index], x_train[dev_sample_index:]
    q_train, q_dev = q_train[:dev_sample_index], q_train[dev_sample_index:]
    y_train, y_dev = y_train[:dev_sample_index], y_train[dev_sample_index:]
    training_size = y_train.shape[0]

    # initialize TensorFlow graph
    graph = tf.get_default_graph()
    with graph.as_default():

        # configurate TensorFlow session, enable GPU accelerated if possible
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            # initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            print("start time:", datetime.now())
            # create model root path if not exists
            if not os.path.exists(model_path):
                os.mkdir(model_path)

            # get current epoch
            curr_epoch = sess.run(model.epoch_step)
            for epoch in range(curr_epoch, num_epochs):
                print("Epoch", epoch + 1, "...")
                counter = 0

                # loop batch training
                for start, end in zip(
                    range(0, training_size, batch_size),
                    range(batch_size, training_size, batch_size),
                ):
                    epoch_x = x_train[start:end]
                    epoch_q = q_train[start:end]
                    epoch_y = y_train[start:end]

                    # create model inputs
                    feed_dict = {
                        model.input_x: epoch_x,
                        model.input_q: epoch_q,
                        model.input_y: epoch_y,
                    }

                    # train one step
                    curr_loss, _ = sess.run(
                        [model.loss_val, model.train_op], feed_dict
                    )
                    counter = counter + 1

                    # evaluation
                    if counter % evaluate_every == 0:
                        train_accu = model.auc.eval(feed_dict)
                        dev_accu = _do_eval(
                            sess, model, x_dev, q_dev, y_dev, batch_size
                        )
                        print(
                            "Step:",
                            counter,
                            "\tLoss:",
                            curr_loss,
                            "\tTraining accuracy:",
                            train_accu,
                            "\tDevelopment accuracy:",
                            dev_accu,
                        )
                sess.run(model.epoch_increment)

                # write model into disk at the end of each epoch
                saver.save(sess, model_path + "/model")
                print("=" * 50)

            print("End time:", datetime.now())


def test_rnn(model_path, prob_norm="softmax", just_graph=False, **kwargs):
    """
    Testing module for BiLSTM models
    
    Parameters
    ----------
    model_path: str
        the path to store the model

    prob_norm: str, default 'softmax'
        method to convert final layer scores into probabilities, either 'softmax' or 'sigmoid'

    just_graph: boolean, default False
        if False, just return tf graphs; if True, take input test data and return y_pred

    **kwargs: dict, optional
        Keyword arguments for test module, full documentation of parameters can be found in notes

    Notes
    ----------
    If just_graph is False, test_rnn should take input test data as follows:

    x_test: 4-D numpy array, shape (num_exemplars, num_bucket, num_section, max_token_length)
        variable indices all buckets and sections

    q_test: 4-D numpy array, shape (num_exemplars, num_bucket, num_section, max_token_length)
        quantities or time deltas of corresponding x_test

    Returns
    ----------
    If just_graph=True:

        sess: tf.Session object
            tf Session for running TensorFlow operations

        t: tf.placeholder
            placeholder for t_test

        x: tf.placeholder
            placeholder for x_test

        q: tf.placeholder
            placeholder for q_test

        y_pred: tf.placeholder
            placeholder for t_test

    If just_graph=False:

        y_probs: 1-D numpy array, shape (num_exemplar,)
            predicted target values based on trained model

    Examples
    --------
    >>> from sheldon.tlstm import test_rnn
    >>> sess, x, q, y_pred = test_rnn('./plstm_model', prob_norm='sigmoid', just_graph=True)
    >>> sess.run(y_pred, {x: x_test, q: q_test})
    array([[4.8457133e-03, 9.9515426e-01],
           [4.6948572e-03, 9.9530518e-01],
           [3.1738445e-02, 9.6826160e-01],
           ...,
           [1.0457519e-03, 9.9895418e-01],
           [5.6348788e-04, 9.9943644e-01],
           [5.9802778e-04, 9.9940193e-01]], dtype=float32)
    >>> sess.close()

    >>> from sheldon.tlstm import test_rnn
    >>> test_rnn('./plstm_model', t_test=T_test, x_test=X_test, q_test=Q_test)
    array([9.9515426e-01,
           4.6948572e-03,
           3.1738445e-02,,
           ...,
           9.9895418e-01,
           5.6348788e-04,
           9.9940193e-01], dtype=float32)
    """

    # clear TensorFlow graph
    tf.reset_default_graph()
    sess = tf.Session()
    sess.as_default()

    # Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph(
        os.path.join(model_path.rstrip("/"), "model.meta")
    )
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    graph = tf.get_default_graph()

    # restore graph names for predictions
    y_score = graph.get_tensor_by_name("output/scores:0")
    assert prob_norm in [
        "softmax",
        "sigmoid",
    ], 'AttributeError: prob_norm only acccept "softmax" or "sigmoid", got {}'.format(
        str(prob_norm)
    )
    if prob_norm == "softmax":
        y_pred = tf.nn.softmax(y_score)
    if prob_norm == "sigmoid":
        y_pred = tf.sigmoid(y_score)
    x = graph.get_tensor_by_name("input_x:0")
    q = graph.get_tensor_by_name("input_q:0")

    if just_graph:
        return sess, x, q, y_pred
    else:
        number_examples = kwargs["y_test"].shape[0]
        y_probs = np.empty((0))
        for start, end in zip(
            range(0, number_examples, 64), range(64, number_examples, 64)
        ):
            feed_dict = {
                x: kwargs["x_test"][start:end],
                q: kwargs["q_test"][start:end],
            }
            probs = sess.run(y_pred, feed_dict)[:, 0]
            y_probs = np.concatenate([y_probs, probs])
        feed_dict = {x: kwargs["x_test"][end:], q: kwargs["q_test"][end:]}
        probs = sess.run(y_pred, feed_dict)[:, 0]
        y_probs = np.concatenate([y_probs, probs])
        sess.close()
        return y_probs


############################# PRIVATE FUNCTIONS ###############################


def _do_eval(sess, model, eval_x, eval_q, eval_y, batch_size):
    """
    Evaluate development in batch (if direcly force sess run entire development set, may raise OOM error)
    """
    number_examples = len(eval_x)
    eval_acc, eval_counter = 0.0, 0
    for start, end in zip(
        range(0, number_examples, batch_size),
        range(batch_size, number_examples, batch_size),
    ):
        feed_dict = {
            model.input_x: eval_x[start:end],
            model.input_q: eval_q[start:end],
            model.input_y: eval_y[start:end],
        }
        curr_eval_acc = model.auc.eval(feed_dict)

        eval_acc, eval_counter = eval_acc + curr_eval_acc, eval_counter + 1
    return eval_acc / float(eval_counter)
