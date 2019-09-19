from functools import partial

from kfp import dsl, components
from kfp.components import InputBinaryFile, OutputBinaryFile

from .common import attach_output_volume

func_to_container_op = partial(
    components.func_to_container_op,
    base_image='rocks.canonical.com:5000/kubeflow/examples/object_detection:latest',
)


@func_to_container_op
def load_task(images: str, annotations: str, records: OutputBinaryFile(str)):
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
def train_task(records: InputBinaryFile(str), pretrained: str, model: OutputBinaryFile(str)):
    from pathlib import Path
    from tensorflow.python.keras.utils import get_file
    import os
    import subprocess
    import shutil
    import re
    import tarfile
    import sys
    from glob import glob

    def load(path):
        return get_file(Path(path).name, path, extract=True)

    model_path = Path(load(pretrained))
    model_path = str(model_path.with_name(model_path.name.split('.')[0]))
    shutil.move(model_path, '/output/model')

    os.mkdir('/checkpoints')

    with tarfile.open(mode='r:gz', fileobj=records) as tar:
        tar.extractall('/output/model')

    print(glob('/output/model/**', recursive=True))
    
    with open('/pipeline.config', 'w') as f:
        config = Path('samples/configs/faster_rcnn_resnet101_pets.config').read_text()
        config = re.sub('PATH_TO_BE_CONFIGURED', '/output/model', config)
        f.write(config)

    shutil.copy('data/pet_label_map.pbtxt', '/output/model/pet_label_map.pbtxt')

    print("Training model")
    subprocess.run(
        [
            sys.executable,
            'model_main.py',
            '--checkpoint_dir',
            '/checkpoints',
            '--model_dir',
            '/output/model',
            '--num_train_steps',
            '1',
            '--pipeline_config_path',
            '/pipeline.config',
            '--run_once=true',
        ],
        check=True
    )

    print(glob('/checkpoints/**', recursive=True))
    print(glob('/output/model/**', recursive=True))
    print(subprocess.check_output(['find', '/', '-name', '*.index']))
    print(subprocess.check_output(['find', '/', '-name', '*.meta']))
    print(subprocess.check_output(['find', '/', '-name', '*ckpt*']))
    print("Exporting inference graph")
    subprocess.run(
        [
            sys.executable,
            'export_inference_graph.py',
            '--input_type',
            'image_tensor',
            '--pipeline_config_path',
            '/pipeline.config',
            '--trained_checkpoint_prefix',
            '/output/model/model.ckpt',
            '--output_directory',
            '/exported_graphs',
        ],
        check=True
    )

    print(glob('/exported_graphs/**', recursive=True))


def serve_sidecar():
    """Serves tensorflow model as sidecar to testing container."""

    return dsl.Sidecar(
        name='tensorflow-serve',
        image='tensorflow/serving:1.14.0',
        command='/usr/bin/tensorflow_model_server',
        args=[
            '--model_name=object_detection',
            '--model_base_path=/output/object_detection',
            '--port=9000',
            '--rest_api_port=9001',
        ],
        mirror_volume_mounts=True,
    )


@func_to_container_op
def test_task(model: InputBinaryFile(str)):
    """Connects to served model and tests example MNIST images."""

    import requests
    from tensorflow.python.keras.backend import get_session
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.saved_model.simple_save import simple_save

    print('Downloaded model, converting it to serving format')
    print(model)
    print(dir(model))

    #      with get_session() as sess:
    #  model = load_model(model.name)
    #  simple_save(
    #      sess,
    #      '/output/mnist/1/',
    #      inputs={'input_image': model.input},
    #      outputs={t.name: t for t in model.outputs},
    #  )

    model_url = 'http://localhost:9001/v1/models/mnist'

    def wait_for_model():
        requests.get('%s/versions/1' % model_url).raise_for_status()

    wait_for_model()

    response = requests.get('%s/metadata' % model_url)
    response.raise_for_status()
    assert response.json() == {}


@dsl.pipeline(name='Object Detection Example', description='')
def object_detection_pipeline(
    images='https://people.canonical.com/~knkski/annotations.tar.gz',
    annotations='https://people.canonical.com/~knkski/images.tar.gz',
    pretrained='https://people.canonical.com/~knkski/faster_rcnn_resnet101_coco_11_06_2017.tar.gz',
):
    loaded = load_task(images, annotations)
    loaded.container.set_gpu_limit(1)
    train = train_task(loaded.outputs['records'], pretrained)
    train.container.set_gpu_limit(1)

    test_task(train.outputs['model']).add_sidecar(serve_sidecar())

    dsl.get_pipeline_conf().add_op_transformer(attach_output_volume)
