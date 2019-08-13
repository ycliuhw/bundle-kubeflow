from kfp import dsl, components
from typing import NamedTuple

from .common import attach_output_volume


def load_task(
    train_images: str, train_labels: str, test_images: str, test_labels: str
) -> NamedTuple('Data', [('bucket', str), ('filename', str)]):
    """Transforms MNIST data from upstream format into numpy array."""

    from gzip import GzipFile
    from pathlib import Path
    from tensorflow.python.keras.utils import get_file
    import numpy as np
    import struct
    import subprocess
    import sys

    try:
        from minio import Minio
    except ModuleNotFoundError:
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'minio'])
        from minio import Minio

    from minio.error import BucketAlreadyOwnedByYou, BucketAlreadyExists

    mclient = Minio('minio:9000', access_key='minio', secret_key='minio123', secure=False)

    bucket = 'mnist'
    filename = 'mnist.npz'

    def load(path):
        return GzipFile(get_file(Path(path).name, path)).read()

    def parse_labels(b: bytes) -> np.array:
        """Parse numeric labels from input data."""
        assert struct.unpack('>i', b[:4])[0] == 0x801
        return np.frombuffer(b[8:], dtype=np.uint8)

    def parse_images(b: bytes) -> np.array:
        """Parse images from input data."""
        assert struct.unpack('>i', b[:4])[0] == 0x803
        count = struct.unpack('>i', b[4:8])[0]
        rows = struct.unpack('>i', b[8:12])[0]
        cols = struct.unpack('>i', b[12:16])[0]

        return np.frombuffer(b[16:], dtype=np.uint8).reshape((count, rows, cols))

    np.savez_compressed(
        f'/output/{filename}',
        **{
            'train_x': parse_images(load(train_images)),
            'train_y': parse_labels(load(train_labels)),
            'test_x': parse_images(load(test_images)),
            'test_y': parse_labels(load(test_labels)),
        },
    )

    try:
        mclient.make_bucket(bucket, location="us-east-1")
    except (BucketAlreadyExists, BucketAlreadyOwnedByYou):
        pass

    mclient.fput_object(bucket, filename, f'/output/{filename}')

    return bucket, filename


def train_task(bucket: str, filename: str) -> NamedTuple('Model', [('filename', str)]):
    """Train CNN on MNIST dataset."""

    import numpy as np
    from tensorflow.python.keras import Sequential
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
    from tensorflow.python.keras.utils import to_categorical
    from tensorflow.python.keras import backend as K
    from tensorflow.python import keras
    import subprocess
    import sys
    from tempfile import TemporaryFile

    try:
        from minio import Minio
    except ModuleNotFoundError:
        subprocess.call([sys.executable, '-m', 'pip', 'install', 'minio'])
        from minio import Minio

    mclient = Minio('minio:9000', access_key='minio', secret_key='minio123', secure=False)

    with TemporaryFile('w+b') as outp:
        with mclient.get_object(bucket, filename) as inp:
            outp.write(inp.read())
        outp.seek(0)
        mnistdata = np.load(outp)

        train_x = mnistdata['train_x']
        train_y = to_categorical(mnistdata['train_y'])
        test_x = mnistdata['test_x']
        test_y = to_categorical(mnistdata['test_y'])

    num_classes = 10
    img_w = 28
    img_h = 28
    batch_size = 128
    epochs = 2

    if K.image_data_format() == 'channels_first':
        train_x.shape = (-1, 1, img_h, img_w)
        test_x.shape = (-1, 1, img_h, img_w)
        input_shape = (1, img_h, img_w)
    else:
        train_x.shape = (-1, img_h, img_w, 1)
        test_x.shape = (-1, img_h, img_w, 1)
        input_shape = (img_h, img_w, 1)

    train_x = train_x[:1000, :, :, :]
    train_y = train_y[:1000, :]

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255
    test_x /= 255

    model = Sequential(
        [
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),
        ]
    )

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'],
    )

    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(test_x, test_y),
    )

    score = model.evaluate(test_x, test_y)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('/output/model.hdf5')

    mclient.fput_object(bucket, 'model.hdf5', '/output/model.hdf5')

    return 'model.hdf5',


@dsl.pipeline(
    name='MNIST CNN Example',
    description='Trains an example Convolutional Neural Network on MNIST dataset.',
)
def mnist_pipeline(
    train_images='https://people.canonical.com/~knkski/train-images-idx3-ubyte.gz',
    train_labels='https://people.canonical.com/~knkski/train-labels-idx1-ubyte.gz',
    test_images='https://people.canonical.com/~knkski/t10k-images-idx3-ubyte.gz',
    test_labels='https://people.canonical.com/~knkski/t10k-labels-idx1-ubyte.gz',
):
    load_op = components.func_to_container_op(
        load_task, base_image='tensorflow/tensorflow:1.14.0-py3'
    )
    train_op = components.func_to_container_op(
        train_task, base_image='tensorflow/tensorflow:1.14.0-py3'
    )
    load = load_op(train_images, train_labels, test_images, test_labels)
    train = train_op(load.outputs['bucket'], load.outputs['filename'])
    train.output_artifact_paths['model'] = '/output/model.hdf5'

    dsl.get_pipeline_conf().add_op_transformer(attach_output_volume)


if __name__ == '__main__':
    data_result = load_task(*mnist_pipeline.__defaults__)[0]
    model_result = train_task(data_result[0], data_result[1])[0]
    print(model_result)
