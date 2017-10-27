import tensorflow as tf

from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras.engine.training import Model
from tensorflow.python.client import device_lib

from models import convolutional, recurrent_l1


def build_model(training_data, model_id, height=28, width=28, multi_gpu=False, gpus=1):
    model = None

    if model_id is convolutional.get_model_id():
        model = convolutional.build(training_data, height=height, width=width)
    elif model_id is recurrent_l1.get_model_id():
        model = recurrent_l1.build(training_data, height=height, width=width)

    if multi_gpu:
        model = _use_multi_gpu(model, gpus=gpus)

    if model:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        print(model.summary())

    return model


def save_model(model, output):
    file_yaml = '{}/model.yaml'.format(output)
    weights_yaml = '{}/model.h5'.format(output)

    model_yaml = model.to_yaml()
    with open(file_yaml, "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, weights_yaml)


def _get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def _normalize_device_name(name):
    name = name.lower().replace('device:', '')
    return name


def _use_multi_gpu(model, gpus):
    if gpus <= 1:
        raise ValueError('For multi-gpu usage to be effective, '
                         'call `multi_gpu_model` with `gpus >= 2`. '
                         'Received: `gpus=%d`' % gpus)

    target_devices = ['/cpu:0'] + ['/gpu:%d' % i for i in range(gpus)]
    available_devices = _get_available_devices()
    available_devices = [_normalize_device_name(name) for name in available_devices]
    for device in target_devices:
        if device not in available_devices:
            raise ValueError(
                'To call `multi_gpu_model` with `gpus=%d`, '
                'we expect the following devices to be available: %s. '
                'However this machine only has: %s. '
                'Try reducing `gpus`.' % (gpus,
                                          target_devices,
                                          available_devices))

    def get_slice(data, i, parts):
        shape = tf.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == gpus - 1:
            size = batch_size - step * i
        else:
            size = step
        size = tf.concat([size, input_shape], axis=0)
        stride = tf.concat([step, input_shape * 0], axis=0)
        start = stride * i
        return tf.slice(data, start, size)

    all_outputs = []
    for i in range(len(model.outputs)):
        all_outputs.append([])

    # Place a copy of the model on each GPU,
    # each getting a slice of the inputs.
    for i in range(gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('replica_%d' % i):
                inputs = []
                # Retrieve a slice of the input.
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_i = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'i': i,
                                                'parts': gpus})(x)
                    inputs.append(slice_i)

                # Apply model on slice
                # (creating a model replica on the target device).
                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save the outputs for merging back together later.
                for o in range(len(outputs)):
                    all_outputs[o].append(outputs[o])

    # Merge outputs on CPU.
    with tf.device('/cpu:0'):
        merged = []
        for outputs in all_outputs:
            merged.append(concatenate(outputs,
                                      axis=0))
        return Model(model.inputs, merged)
