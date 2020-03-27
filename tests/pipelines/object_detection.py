from functools import partial

from kfp import dsl, components
from kfp.components import InputBinaryFile, OutputBinaryFile

from .common import attach_output_volume

func_to_container_op = partial(
    components.func_to_container_op,
    base_image='rocks.canonical.com:5000/kubeflow/examples/object_detection:latest',
)


@func_to_container_op
def load_task(
    images: str,
    annotations: str,
    records: OutputBinaryFile(str),
    validation_images: OutputBinaryFile(str),
):
    """Transforms MNIST data from upstream format into numpy array."""

    from glob import glob
    from pathlib import Path
    from tensorflow.python.keras.utils import get_file
    import subprocess
    import tarfile

    def load(path):
        return get_file(Path(path).name, path, extract=True)

    load(images)
    load(annotations)

    with tarfile.open(mode='w:gz', fileobj=validation_images) as tar:
        for image in glob('/root/.keras/datasets/images/*.jpg')[:10]:
            tar.add(image, arcname=Path(image).name)

    subprocess.run(
        [
            'python',
            'object_detection/dataset_tools/create_pet_tf_record.py',
            '--label_map_path=object_detection/data/pet_label_map.pbtxt',
            '--data_dir',
            '/root/.keras/datasets/',
            '--output_dir=/models/research',
        ],
        check=True,
        cwd='/models/research',
    )

    with tarfile.open(mode='w:gz', fileobj=records) as tar:
        for record in glob('/models/research/*.record-*'):
            tar.add(record, arcname=Path(record).name)


@func_to_container_op
def train_task(records: InputBinaryFile(str), pretrained: str, exported: OutputBinaryFile(str)):
    from pathlib import Path
    from tensorflow.python.keras.utils import get_file
    import subprocess
    import shutil
    import re
    import tarfile
    import sys

    def load(path):
        return get_file(Path(path).name, path, extract=True)

    model_path = Path(load(pretrained))
    model_path = str(model_path.with_name(model_path.name.split('.')[0]))
    shutil.move(model_path, '/model')

    with tarfile.open(mode='r:gz', fileobj=records) as tar:
        tar.extractall('/records')

    with open('/pipeline.config', 'w') as f:
        config = Path('samples/configs/faster_rcnn_resnet101_pets.config').read_text()
        config = re.sub(r'PATH_TO_BE_CONFIGURED\/model\.ckpt', '/model/model.ckpt', config)
        config = re.sub('PATH_TO_BE_CONFIGURED', '/records', config)
        f.write(config)

    shutil.copy('data/pet_label_map.pbtxt', '/records/pet_label_map.pbtxt')

    print("Training model")
    subprocess.run(
        [
            sys.executable,
            'model_main.py',
            '--model_dir',
            '/model',
            '--num_train_steps',
            '1',
            '--pipeline_config_path',
            '/pipeline.config',
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            'export_inference_graph.py',
            '--input_type',
            'image_tensor',
            '--pipeline_config_path',
            '/pipeline.config',
            '--trained_checkpoint_prefix',
            '/model/model.ckpt-1',
            '--output_directory',
            '/exported',
        ],
        check=True,
    )

    with tarfile.open(mode='w:gz', fileobj=exported) as tar:
        tar.add('/exported', recursive=True)


def serve_sidecar():
    """Serves tensorflow model as sidecar to testing container."""

    return dsl.Sidecar(
        name='tensorflow-serve',
        image='tensorflow/serving:1.15.0-gpu',
        command='/usr/bin/tensorflow_model_server',
        args=[
            '--model_name=object_detection',
            '--model_base_path=/exported',
            '--port=9000',
            '--rest_api_port=9001',
        ],
        mirror_volume_mounts=True,
    )


@func_to_container_op
def test_task(model: InputBinaryFile(str), validation_images: InputBinaryFile(str)):
    """Connects to served model and tests example MNIST images."""

    import tarfile
    import requests
    import time
    import tensorflow as tf
    from tensorflow.saved_model import load_v2

    print('Downloaded model, converting it to serving format')
    print(model)
    print(dir(model))

    with tarfile.open(model.name) as tar:
        tar.extractall(path="/")

    with tarfile.open(validation_images.name) as tar:
        tar.extractall(path="/images")

    import glob

    print(glob.glob('/images/*'))
    loaded = load_v2('/exported/saved_model')
    print(loaded)
    print(type(loaded))
    print(dir(loaded))
    print(loaded.asset_paths)
    print(loaded.graph)
    print(loaded.initializer)
    print(loaded.signatures)
    print(loaded.signatures['serving_default'])
    print(loaded.tensorflow_version)
    print(loaded.variables)
    infer = loaded.signatures['serving_default']
    print(infer(tf.cast(tf.fill((100, 100, 5, 3), 0), tf.uint8)))

    model_url = 'http://localhost:9001/v1/models/object_detection'
    for _ in range(60):
        try:
            requests.get(f'{model_url}/versions/1').raise_for_status()
            break
        except requests.RequestException:
            time.sleep(5)
    else:
        raise Exception("Waited too long for sidecar to come up!")

    #  response = requests.get('%s/metadata' % model_url)
    #  response.raise_for_status()
    #  assert response.json() == {}


@dsl.pipeline(name='Object Detection Example', description='')
def object_detection_pipeline(
    images='https://people.canonical.com/~knkski/images.tar.gz',
    annotations='https://people.canonical.com/~knkski/annotations.tar.gz',
    pretrained='https://people.canonical.com/~knkski/faster_rcnn_resnet101_coco_11_06_2017.tar.gz',
):
    loaded = load_task(images, annotations)
    loaded.container.set_gpu_limit(1)
    train = train_task(loaded.outputs['records'], pretrained)
    train.container.set_gpu_limit(1)

    test_task(train.outputs['exported'], loaded.outputs['validation_images']).add_sidecar(serve_sidecar())

    dsl.get_pipeline_conf().add_op_transformer(attach_output_volume)
