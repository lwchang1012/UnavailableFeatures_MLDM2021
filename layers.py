import tensorflow as tf


class DropoutDense(tf.keras.layers.Dense):
    def __init__(
            self,
            units,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            dropout_dims=None,
            dropout_rate_fn=None,
            dropout_mode='random_drop',
            **kwargs):
        super(DropoutDense, self).__init__(
                units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs)
        self.dropout_dims = dropout_dims
        self.dropout_rate_fn = dropout_rate_fn
        self.dropout_mode = dropout_mode

    def build(self, input_shape):
        super(DropoutDense, self).build(input_shape)
        self.step = self.add_weight(
                'step',
                shape=[],
                dtype=tf.int64,
                initializer='zeros',
                trainable=False)
        self.orig_kernel = self.kernel

    def get_masked_kernel(self, training=None):
        if self.dropout_rate_fn is not None and training is True:
            step = tf.cast(self.step, self.dtype)
            rate = self.dropout_rate_fn(step)
        else:
            rate = 1.0

        if self.dropout_dims:
            dropout_dims = [[i] for i in self.dropout_dims]

            if self.dropout_mode == 'random_drop':
                dropout_mask = tf.random.uniform([len(self.dropout_dims), self.units])
                dropout_mask = tf.cast(tf.less_equal(dropout_mask, rate), self.dtype)
            if self.dropout_mode == 'multiply_drop':
                dropout_mask = tf.ones([len(self.dropout_dims), self.units])
                dropout_mask = dropout_mask * rate

            kernel_mask = tf.tensor_scatter_nd_sub(
                    tf.ones_like(self.orig_kernel),
                    dropout_dims,
                    dropout_mask)
            kernel = self.orig_kernel * kernel_mask
        else:
            kernel = self.orig_kernel

        return kernel

    def call(self, inputs, training=None):
        if training is True:
            self.step.assign_add(1)

        self.kernel = self.get_masked_kernel(training=training)

        return super(DropoutDense, self).call(inputs)

    def get_config(self):
        config = super(DropoutDense, self).get_config()
        config['dropout_dims'] = self.dropout_dims
        config['dropout_mode'] = self.dropout_mode
        return config
