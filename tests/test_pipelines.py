import time
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Callable

import pytest
from kfp.compiler import Compiler

from .pipelines.cowsay import cowsay_pipeline
from .pipelines.mnist import mnist_pipeline
from .utils import get_session, get_pub_addr


COWSAY_PARAMS = [{"name": "url", "value": "https://helloacm.com/api/fortune/"}]


MNIST_PARAMS = [
    {"name": "test-images", "value": "https://people.canonical.com/~knkski/t10k-images-idx3-ubyte.gz"},
    {"name": "test-labels", "value": "https://people.canonical.com/~knkski/t10k-labels-idx1-ubyte.gz"},
    {"name": "train-images", "value": "https://people.canonical.com/~knkski/train-images-idx3-ubyte.gz"},
    {"name": "train-labels", "value": "https://people.canonical.com/~knkski/train-labels-idx1-ubyte.gz"},
]


@pytest.mark.parametrize('name,params,fn', [
    ('mnist', MNIST_PARAMS, mnist_pipeline),
    ('cowsay', COWSAY_PARAMS, cowsay_pipeline),
])
def test_mnist(name: str, params: list, fn: Callable):
    sess = get_session()
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    endpoint = f'http://{get_pub_addr()}/pipeline/apis/v1beta1'
    pipeline_name = f'{name}-{now}'
    run_name = f'{name} {now}'
    run_description = f"Automated testing run of pipeline {pipeline_name}"

    with NamedTemporaryFile('w+b', suffix='.tar.gz') as f:
        Compiler().compile(fn, f.name)
        pipeline = sess.post(
            f'{endpoint}/pipelines/upload', files={'uploadfile': f}, params={'name': pipeline_name}
        ).json()

        assert pipeline['name'] == pipeline_name
        assert sorted(pipeline['parameters'], key=lambda x: x['name']) == params

        pl_run = sess.post(
            f'{endpoint}/runs',
            json={
                "description": run_description,
                "name": run_name,
                "pipeline_spec": {
                    "parameters": params,
                    "pipeline_id": pipeline['id'],
                },
                "resource_references": [],
            },
        ).json()['run']

        for _ in range(24):
            pl_run = sess.get(f'{endpoint}/runs/{pl_run["id"]}').json()['run']
            status = pl_run.get('status')
            if status == 'Failed':
                pytest.fail(f"Pipeline run encountered an error: {pl_run}")
            elif status != 'Succeeded':
                print("Waiting for pipeline to finish.")
                time.sleep(5)
            else:
                assert status == 'Succeeded'
                break
        else:
            pytest.fail("Waited too long for pipeline to succeed!")
