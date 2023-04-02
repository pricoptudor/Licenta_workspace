import random
from confugue import configurable
import numpy as np
import tensorflow as tf
import re
from note_seq.protobuf import music_pb2
import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay 
from keras.optimizers import Adam

class DatasetManager:
    """A class for managing TensorFlow datasets.

    A `DatasetManager` objects holds a collection of datasets (typically, a training set and a
    validation set) and their iterators and takes care of switching between them when running
    TensorFlow ops.
    """

    def __init__(self, output_types=None, output_shapes=None):
        self._output_types = output_types
        self._output_shapes = output_shapes

        self.datasets = {}
        self._iterators = {}
        self._handles = {}
        self._handle_placeholder = tf.Variable("", dtype=tf.string, name='dataset_handle')
        self._global_iterator = None
        self._last_session = None

    def add_dataset(self, name, dataset, one_shot=False):
        """Add a new dataset to the collection.

        Args:
            name: A name for the dataset (e.g. `'train'`, `'val'`).
            dataset: An instance of `tf.data.Dataset`.
            one_shot: If `True`, a one-shot iterator will be used (appropriate for training
                datasets); if `False` (default), an initializable iterator will be used
                (requiring to call `initialize_dataset`).
        """
        if name in self.datasets:
            self.remove_dataset(name)

        self.datasets[name] = dataset
        if one_shot:
            self._iterators[name] = iter(dataset)
        else:
            self._iterators[name] = iter(dataset)

        if self._output_types is None:
            self._output_types = dataset.output_types
        if self._output_shapes is None:
            self._output_shapes = dataset.output_shapes

    def remove_dataset(self, name):
        """Remove the given dataset from the collection."""
        del self.datasets[name]
        del self._iterators[name]
        del self._handles[name]

    def initialize_dataset(self, session, name):
        """Initialize the given dataset's iterator."""
        # session.run(self._iterators[name].initializer)

    def run(self, session, fetches, dataset_name=None, feed_dict=None, options=None):
        """Run the given TensorFlow ops while using the chosen dataset.

        Args:
            session: A TensorFlow `Session`.
            fetches: The `fetches` to pass to `session.run`.
            dataset_name: The name of the dataset to use. If `None` (default), no dataset will be
                used.
            feed_dict: The `feed_dict` to pass to `session.run`.
            options: A `RunOptions` proto to pass to `session.run`.
        Returns:
            The return value of `session.run`.
        """
        if feed_dict is None:
            feed_dict = {}
        if dataset_name is not None:
            if session is not self._last_session:
                self._handles.clear()

            if dataset_name not in self._handles:
                iterator = self._iterators[dataset_name]
                self._handles[dataset_name] = iterator.string_handle()
            feed_dict[self._handle_placeholder] = self._handles[dataset_name]
        self._last_session = session
        return session.run(fetches, feed_dict, options=options)

    def run_over_dataset(self, session, fetches, dataset, feed_dict=None, concat_batches=False,
                         options=None):
        """Run the given TensorFlow ops while iterating over an entire dataset.

        Args:
            session: A TensorFlow `Session`.
            fetches: The `fetches` to pass to `session.run`.
            dataset: The name of the dataset to use, or a new `tf.data.Dataset`.
            feed_dict: The `feed_dict` to pass to `session.run`.
            concat_batches: If `True`, the results will be concatenated along the first dimension.
            options: A `RunOptions` proto to pass to `session.run`.
        Returns:
            A list of results of `session.run`. If `fetches` is a nested structure of tensors,
            then the same nested structure will be returned, containing a list of results for each
            tensor. If `concat_batches` is `True`, the return value will be a list (or a nested
            structure of lists) obtained by concatenating all batches.
        """
        if isinstance(dataset, str):
            dataset_name = dataset
        else:
            # A dataset object was passed directly; add it temporarily, then remove it.
            dataset_name = '__tmp'
            self.add_dataset(dataset_name, dataset)

        self.initialize_dataset(session, dataset_name)
        results = []
        while True:
            try:
                results.append(self.run(session, fetches, dataset_name, feed_dict=feed_dict,
                                        options=options))
            except tf.errors.OutOfRangeError:
                break

        if not results:
            return None

        # Flatten the structure of each batch, put the corresponding elements together and restore
        # the structure.
        structure = results[0]
        results_flat = zip(*(tf.nest.flatten(r) for r in results))
        if concat_batches:
            # We do not use np.concatenate since the shapes of the batches might be incompatible.
            # Instead, we stack the items of all batches in a list.
            results_flat = [[x for batch in r for x in batch] for r in results_flat]
        else:
            results_flat = [list(r) for r in results_flat]
        results = tf.nest.pack_sequence_as(structure, results_flat)

        if dataset_name == '__tmp':
            self.remove_dataset(dataset_name)

        return results

    def get_next(self):
        """Return a nested struture of tensors representing the next element of a dataset."""
        if self._global_iterator is None:
            self._global_iterator = tf.data.Iterator.from_string_handle(
                self._handle_placeholder, self._output_types, self._output_shapes)
        return self._global_iterator.get_next()

    def get_batch(self):
        """Return a nested struture of tensors representing the next element of a dataset."""
        return self.get_next()

def create_train_op(loss, optimizer=None, variables=None, max_gradient_norm=None,
                    name='training'):
    """Create a training op."""
    global_step = 0

    if optimizer is None:
        learning_rate = ExponentialDecay(initial_learning_rate=0.001,
                                    decay_steps=3000,
                                    decay_rate=0.5)
        optimizer = Adam(learning_rate=learning_rate)

        if learning_rate is not None:
            tf.summary.scalar('learning_rate', learning_rate, family='train')

    if variables is None:
        module = tf.Module()
        variables = module.trainable_variables

    print("'{}' op trains {} variables:\n\n{}\n".format(
        name, len(variables), summarize_variables(variables)))

    grads_and_vars = optimizer.compute_gradients(loss, variables)
    return optimizer.apply_gradients(
        clip_gradients(grads_and_vars, max_gradient_norm),
        global_step=global_step)

def clip_gradients(grads_and_vars, max_gradient_norm):
    """Perform gradient clipping by global norm."""
    if max_gradient_norm is None:
        return grads_and_vars

    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    return zip(clipped_gradients, variables)

def summarize_variables(variables):
    """Return a string with a table of names and shapes of the given variables."""
    cols = [['Name'], ['Shape']]
    for var in variables:
        shape = var.get_shape().as_list()
        cols[0].append(var.name)
        cols[1].append(str(shape))
    widths = [max(len(x) for x in col) for col in cols]

    lines = ['  '.join(text.ljust(widths[i]) for i, text in enumerate(row)) for row in zip(*cols)]
    lines.insert(1, '  '.join('-' * w for w in widths))
    return '\n'.join(lines)

def make_simple_dataset(generator, output_types, output_shapes, batch_size=None, name='dataset',
                        preprocess_fn=None):
    """Create a simple validation or test dataset."""
    with tf.name_scope(name):
        dataset = tf.data.Dataset.from_generator(generator, output_types)
        if preprocess_fn:
            dataset = dataset.map(preprocess_fn)
        if batch_size is not None:
            dataset = dataset.padded_batch(batch_size, output_shapes)
        return dataset
    
def prepare_train_and_val_data(train_generator, val_generator, output_types, output_shapes,
                               train_batch_size, val_batch_size, shuffle_buffer_size=100000,
                               preprocess_fn=None, num_epochs=None, num_train_examples=None,
                               dataset_manager=None):
    """Prepare a DatasetManager with training and validation data.

    Args:
        train_generator: A generator yielding the training examples.
        val_generator: A generator yielding the validation examples.
        output_types: The type(s) of the elements of the dataset.
        output_shapes: The padded shape(s) of the elements of the dataset.
        train_batch_size: The batch size of the training dataset.
        val_batch_size: The batch size of the validation dataset.
        shuffle_buffer_size: The size of the buffer used for sampling elements from the training
            dataset.
        preprocess_fn: The pre-processing function to apply to the data.
        num_epochs: The number of training epochs. If `None`, the training dataset will loop
            indefinitely.
        num_train_examples: If given, the number of examples per training epoch will be limited
            to this number (before shuffling).
        dataset_manager: The `DatasetManager` to add the datasets to.

    Return:
        A tuple `(train_dataset, val_dataset)`.
    """
    train_dataset = make_train_dataset(train_generator, output_types, output_shapes,
                                       train_batch_size, shuffle_buffer_size, preprocess_fn,
                                       num_epochs, num_train_examples)
    if dataset_manager:
        dataset_manager.add_dataset('train', train_dataset, one_shot=True)

    val_dataset = make_simple_dataset(
        val_generator, output_types, output_shapes, val_batch_size, 'val',
        preprocess_fn=preprocess_fn)
    if dataset_manager:
        dataset_manager.add_dataset('val', val_dataset)

    return train_dataset, val_dataset

def make_train_dataset(generator, output_types, output_shapes, batch_size,
                       shuffle_buffer_size=100000, preprocess_fn=None, num_epochs=None,
                       num_examples=None, name='train'):
    """Prepare a training dataset.

    Args:
        generator: A generator yielding the examples.
        output_types: The type(s) of the elements of the dataset.
        output_shapes: The padded shape(s) of the elements of the dataset.
        batch_size: The batch size of the dataset.
        shuffle_buffer_size: The size of the buffer used for sampling elements from the dataset.
        preprocess_fn: The pre-processing function to apply to the data.
        num_epochs: The number of training epochs. If `None`, the training dataset will loop
            indefinitely.
        num_examples: If given, the number of examples per training epoch will be limited
            to this number (before shuffling).
        name: A name for the name scope for the dataset.

    Return:
        A `tf.data.Dataset`.
    """
    with tf.name_scope(name):
        dataset = tf.data.Dataset.from_generator(generator, output_types)
        if num_examples:
            dataset = dataset.take(num_examples)
        if shuffle_buffer_size:
            dataset = dataset.shuffle(shuffle_buffer_size,
                                      reshuffle_each_iteration=True)
        if preprocess_fn:
            dataset = dataset.map(preprocess_fn)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.padded_batch(batch_size, output_shapes)
        return dataset

def set_random_seed(seed):
    if seed is not None:
        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


def filter_sequence(sequence, instrument_re=None, instrument_ids=None, programs=None, drums=None,
                    copy=False):
    """Filter a Magenta `NoteSequence` in place.

    Args:
        sequence: The `NoteSequence` protobuffer to filter.
        instrument_re: A regular expression used to match instrument names.
        instrument_ids: A list of instrument IDs to match or `None` to match any ID.
        programs: A list of MIDI programs to match or `None` to match any program.
        drums: Include only drums (`True`) or only non-drums (`False`). If `None` (default), include
            both drums and non-drums.
        copy: If `True`, a copy of the sequence will be returned and the original sequence will
            be left unchanged.

    Returns:
        The filtered `NoteSequence`.
    """
    if copy:
        sequence, original_sequence = music_pb2.NoteSequence(), sequence
        sequence.CopyFrom(original_sequence)

    if isinstance(instrument_re, str):
        instrument_re = re.compile(instrument_re)

    # Filter the instruments based on name and ID
    deleted_ids = set()
    if instrument_re is not None:
        deleted_ids.update(i.instrument for i in sequence.instrument_infos
                           if not instrument_re.search(i.name))
    if instrument_ids is not None:
        deleted_ids.update(i.instrument for i in sequence.instrument_infos
                           if i.instrument not in instrument_ids)
    new_infos = [i for i in sequence.instrument_infos if i.instrument not in deleted_ids]
    del sequence.instrument_infos[:]
    for info in new_infos:
        sequence.instrument_infos.add().CopyFrom(info)

    # Filter the event collections
    for collection in [sequence.notes, sequence.pitch_bends, sequence.control_changes]:
        collection_copy = list(collection)
        del collection[:]

        for event in collection_copy:
            if event.instrument in deleted_ids:
                continue
            if instrument_ids is not None and event.instrument not in instrument_ids:
                continue
            if programs is not None and event.program not in programs:
                continue
            if drums is not None and event.is_drum != drums:
                continue
            collection.add().CopyFrom(event)

    return sequence

def set_note_fields(sequence, **kwargs):
    for note in sequence.notes:
        for attr, val in kwargs.items():
            setattr(note, attr, val)















