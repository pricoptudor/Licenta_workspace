import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dropout,\
                            GRUCell, RNN, Bidirectional, Dense
from tensorflow_addons.seq2seq import TrainingSampler, GreedyEmbeddingSampler, SampleEmbeddingSampler,\
                                        BasicDecoder, dynamic_decode
    
class EmbeddingLayer():
    def __init__(self, input_size, output_size, name='embedding'):
        self.name = name
        self.input_size = input_size
        self.output_size = output_size

        self.embedding_matrix = tf.get_variable(
            'embedding_matrix', shape=[self.input_size, self.output_size])

    def embed(self, x):
        return tf.nn.embedding_lookup(self.embedding_matrix, x)

    def __call__(self, inputs):
        return self.embed(inputs)

class DropoutWrapper(tf.nn.rnn_cell.DropoutWrapper):  # pylint: disable=abstract-method
    """A version of `tf.nn.rnn_cell.DropoutWrapper` that disables dropout during inference."""

    def __init__(self, cell, training, **kwargs):
        """Initialize the wrapper.

        Args:
            cell: An `RNNCell`.
            training: A `tf.bool` tensor indicating whether we are in training mode.
            **kwargs: Any other arguments to `tf.nn.rnn_cell.DropoutWrapper`.
        """
        for key in ['input_keep_prob', 'output_keep_prob', 'state_keep_prob']:
            if key in kwargs:
                kwargs[key] = tf.cond(training,
                                      lambda key=key: tf.convert_to_tensor(kwargs[key]),
                                      lambda: tf.constant(1.))

        super().__init__(cell, **{'dtype': tf.float32, **kwargs})

class InputWrapper(tf.nn.rnn_cell.RNNCell):
    """A wrapper for passing additional input to an RNN cell."""

    def __init__(self, cell, input_fn):
        """Initialize the wrapper.

        Args:
            cell: An `RNNCell`.
            input_fn: A function expecting a scalar tensor argument `batch_size` and returning
                a tensor of shape `[batch_size, input_size]` to concatenate with the RNN cell input.
        """
        super().__init__()
        self._cell = cell
        self._dtype = self._cell.dtype
        self._input_fn = input_fn

    @property
    def wrapped_cell(self):
        return self._cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        return self._cell.compute_output_shape(input_shape)

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        batch_size = tf.shape(state)[0]
        inputs = tf.concat([inputs, self._input_fn(batch_size)], axis=-1)
        return self._cell(inputs, state, scope=scope)


class RNNDecoder():

    def __init__(self,
                 vocabulary,
                 embedding_layer,
                 attention_mechanism=None,
                 pre_attention=False,
                 max_length=None,
                 cell=None,
                 cell_wrap_fn=None,
                 output_projection=None,
                 training=None,
                 name='decoder'):
        self.name = name

        self._vocabulary = vocabulary
        self._embeddings = embedding_layer
        self._attention_mechanism = attention_mechanism
        self._max_length = max_length
        self._training = training

        
        if not cell:
            cell = GRUCell(units=1024, dtype=tf.float32)
        if cell_wrap_fn:
            cell = cell_wrap_fn(cell)
        self._dtype = cell.dtype
        self.initial_state_size = cell.state_size

        cell_dropout = DropoutWrapper(cell=cell,
                                      dtype=tf.float32,
                                      training=self._training)
        self.cell = cell_dropout or cell

        if self._attention_mechanism:
            self.cell = _AttentionWrapper(cell=self.cell,
                                          attention_mechanism=self._attention_mechanism,
                                          output_attention=False,
                                          pre_attention=pre_attention,
                                          input_size=self._embeddings.output_size)
        self.cell.build(tf.TensorShape([None, self._embeddings.output_size]))

        if output_projection:
            self._output_projection = output_projection
        else:
            self._output_projection = Dense(units=len(vocabulary), 
                                            use_bias=False,
                                            name='output_projection')
            
        self._output_projection.build([None, self.cell.output_size])
        self._built = True

    def decode_train(self, inputs, targets, initial_state=None):
        target_weights = tf.sign(targets, name='target_weights')
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]

        embedded_inputs = self._embeddings.embed(inputs)

        # Apply token dropout if defined in the configuration. This replaces embeddings at random
        # positions with zeros.
        dropped_inputs = Dropout(inputs=embedded_inputs, 
                                 noise_shape=[batch_size, inputs_shape[1], 1],
                                 training=self._training,
                                 name='token_dropout')
        
        if dropped_inputs is not None:
            embedded_inputs = dropped_inputs

        with tf.name_scope('decode_train'):
            initial_state = self._make_initial_state(batch_size, initial_state)
            sequence_length = tf.reduce_sum(target_weights, axis=1)
            helper = TrainingSampler(inputs=embedded_inputs,
                                     sequence_length=sequence_length,
                                     time_major=False)
            output, _ = self._dynamic_decode(helper=helper,
                                             initial_state=initial_state)
            logits = output.rnn_output

        with tf.name_scope('loss'):
            batch_size = tf.shape(logits)[0]
            train_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets,
                logits=logits)
            loss = (tf.reduce_sum(train_xent * tf.to_float(target_weights)) /
                    tf.to_float(batch_size))

        return output, loss

    def decode(self, initial_state=None, max_length=None, batch_size=None,
               softmax_temperature=1., random_seed=None, mode='greedy'):
        with tf.name_scope('decode_{}'.format(mode)):
            if batch_size is None:
                batch_size = tf.shape(initial_state)[0]
            initial_state = self._make_initial_state(batch_size, initial_state)
            helper = self._make_helper(batch_size, softmax_temperature, random_seed, mode)

            return self._dynamic_decode(helper=helper,
                                        initial_state=initial_state,
                                        max_length=max_length or self._max_length)

    def _make_initial_state(self, batch_size, cell_state=None):
        if cell_state is None:
            return self.cell.zero_state(batch_size, dtype=self._dtype)

        if self._attention_mechanism:
            # self.cell is an instance of AttentionWrapper. We need to get its zero_state and
            # replace the cell state wrapped in it.
            wrapper_state = self.cell.zero_state(batch_size, dtype=self._dtype)
            return wrapper_state.clone(cell_state=cell_state)

        return cell_state

    def _make_helper(self, batch_size, softmax_temperature, random_seed, mode):
        helper_kwargs = {
            'embedding': self._embeddings.embedding_matrix,
            'start_tokens': tf.tile([self._vocabulary.start_id], [batch_size]),
            'end_token': self._vocabulary.end_id
        }

        if mode == 'greedy':
            return GreedyEmbeddingSampler(**helper_kwargs)
        if mode == 'sample':
            helper_kwargs['softmax_temperature'] = softmax_temperature
            helper_kwargs['seed'] = random_seed
            return SampleEmbeddingSampler(**helper_kwargs)

        raise ValueError('Unrecognized mode {!r}'.format(mode))

    def _dynamic_decode(self, helper, initial_state, max_length=None):
        decoder = BasicDecoder(
            cell=self.cell,
            helper=helper,
            initial_state=initial_state,
            output_layer=self._output_projection)
        output, state, _ = dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_length)
        return output, state

class _AttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    """A modified `AttentionWrapper`.

    This wrapper adds an attention step before starting the decoding (if enabled by the
    `pre_attention` argument). This is necessary if we don't pass an initial state.
    """

    def __init__(self, *args, pre_attention=False, input_size=None, **kwargs):
        self._pre_attention = pre_attention
        self._input_size = input_size
        super().__init__(*args, **kwargs)

    def zero_state(self, batch_size, dtype):
        zero_state = super().zero_state(batch_size, dtype)
        if not self._pre_attention:
            return zero_state

        # Do one step (to get the attention running), but revert the RNN cell
        # back to the zero state.
        inputs = tf.zeros([batch_size, self._input_size], dtype)
        _, new_state = self.__call__(inputs, zero_state)
        return new_state.clone(cell_state=zero_state.cell_state)

_AttentionWrapper.__name__ = 'AttentionWrapper'  # needed for nice TensorFlow variable scope name



class RNNLayer():

    def __init__(self, training=None, forward_cell=None, backward_cell=None,
                 output_states='all', name='rnn'):
        self.name = name

        if output_states not in ['all', 'output', 'final']:
            raise ValueError(f"Invalid value for output_states: '{output_states}'; "
                             "expected 'output', 'final' or 'all'")
        self._output_states = output_states
        self._training = training

        if forward_cell:
            fw_cell = forward_cell
        else:
            if 'style' not in self.name:
                fw_cell = GRUCell(units=200, dtype=tf.float32)
            else:
                fw_cell = GRUCell(units=500, dtype=tf.float32)
        fw_cell_dropout = DropoutWrapper(cell=fw_cell, training=self._training)
        self._fw_cell = fw_cell_dropout or fw_cell

        if backward_cell:
            self._bw_cell = backward_cell
        else:
            if 'style' not in self.name:
                self._bw_cell = GRUCell(units=200, dtype=tf.float32)
            else:
                self._bw_cell = GRUCell(units=500, dtype=tf.float32)

        if self._bw_cell:
            bw_cell_dropout = DropoutWrapper(cell=self._bw_cell, training=self._training)
            self._bw_cell = bw_cell_dropout or self._bw_cell

        self._final_dropout = Dropout()

    def apply(self, inputs):
        if not self._bw_cell:
            outputs, final_states = RNN(self._fw_cell, inputs=inputs, dtyp=tf.float32,
                                        return_sequences=True, return_state=True)
        else:
            outputs, final_states = Bidirectional(RNN([self._fw_cell, self._bw_cell],
                                                      inputs=inputs, dtype=tf.float32))
            outputs = tf.concat(outputs, -1)
            final_states = tf.concat(final_states, -1)

        if self._final_dropout:
            final_states = self._final_dropout(final_states, training=self._training)

        if self._output_states == 'output':
            return outputs
        elif self._output_states == 'final':
            return final_states
        else:
            return outputs, final_states

    def __call__(self, inputs):
        return self.apply(inputs)


class CNN():
    def __init__(self, training=None, name='cnn'):
        self.name = name

        self._is_training = training

        self._layers_2d = [ 
            Conv2D(filters=32, 
                   kernel_size=(12, 12),
                   padding='same',
                   activation='elu'),
            MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2)),
            Conv2D(filters=32,
                   kernel_size=(4, 4),
                   padding='same',
                   activation='elu'),
            MaxPooling2D(pool_size=(2, 4),
                         strides=(2, 4))
        ]
        self._layers_1d = [
            Conv1D(filters=300,
                   kernel_size=6,
                   padding='same',
                   activation='elu'),
            MaxPooling1D(pool_size=2,
                         strides=2),
            Conv1D(filters=300,
                   kernel_size=4,
                   padding='same',
                   activation='elu'),
            MaxPooling1D(pool_size=2,
                         strides=2),
            Conv1D(filters=300,
                   kernel_size=4,
                   padding='same',
                   activation='elu'),
            MaxPooling1D(pool_size=2,
                         strides=2)
        ]

    def __call__(self, inputs):
        return self.apply(inputs)

    def apply(self, inputs):
        batch_size = tf.shape(inputs)[0]
        features = inputs
        if self._layers_2d:
            if features.shape.ndims == 3:
                # Expand to 4 dimensions: [batch_size, rows, time, channels]
                features = tf.expand_dims(features, -1)

            # 2D layers: 4 -> 4 dimensions
            for layer in self._layers_2d:
                print(f'Inputs to layer {layer} have shape {features.shape}')
                features = self._apply_layer(layer, features)
            print(f'After the 2D layers, the features have shape {features.shape}')

            # Features have shape [batch_size, rows, time, channels]. Switch rows and cols, then
            # flatten rows and channels to get 3 dimensions: [batch_size, time, new_channels].
            features = tf.transpose(features, perm=[0, 2, 1, *range(3, features.shape.ndims)])
            num_channels = features.shape[2] * features.shape[3]
            features = tf.reshape(features, [batch_size, -1, num_channels])

        # 1D layers: 3 -> 3 dimensions: [batch_size, time, channels]
        for layer in self._layers_1d:
            print(f'Inputs to layer {layer} have shape {features.shape}')
            features = self._apply_layer(layer, features)

        return features

    def _apply_layer(self, layer, features):
        if isinstance(layer, Dropout):
            return layer(features, training=self._is_training)
        return layer(features)
