from kfp import dsl
from .common import attach_output_volume


def fortune_task(url):
    """Get a random fortune."""
    return dsl.ContainerOp(
        name='fortune',
        image='appropriate/curl',
        command=['sh', '-c'],
        arguments=['curl $0 | tee $1', url, '/output/fortune.txt'],
        file_outputs={'text': '/output/fortune.txt'},
    )


def cow_task(text):
    """Have a cow say something"""
    return dsl.ContainerOp(
        name='cowsay',
        image='chuanwen/cowsay',
        command=['bash', '-c'],
        arguments=['/usr/games/cowsay "$0" | tee $1', text, '/output/cowsay.txt'],
        file_outputs={'text': '/output/cowsay.txt'},
    )


@dsl.pipeline(name='Fortune Cow', description='Talk to a fortunate cow.')
def cowsay_pipeline(url='https://helloacm.com/api/fortune/'):
    fortune = fortune_task(url)
    cowsay = cow_task(fortune.output)

    dsl.get_pipeline_conf().add_op_transformer(attach_output_volume)
