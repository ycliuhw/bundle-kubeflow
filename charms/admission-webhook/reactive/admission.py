import os
from base64 import b64encode
from pathlib import Path
from subprocess import run

import yaml

from charmhelpers.core import hookenv
from charms import layer
from charms.reactive import clear_flag, hook, set_flag, when, when_not


@hook('upgrade-charm')
def upgrade_charm():
    clear_flag('charm.started')


@when('charm.started')
def charm_ready():
    layer.status.active('')


@when('layer.docker-resource.oci-image.changed')
def update_image():
    clear_flag('charm.started')


@when('layer.docker-resource.oci-image.available')
@when_not('charm.started')
def start_charm():
    layer.status.maintenance('configuring container')

    image_info = layer.docker_resource.get_info('oci-image')
    model = os.environ['JUJU_MODEL_NAME']

    run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:4096",
            "-keyout",
            "key.pem",
            "-out",
            "cert.pem",
            "-days",
            "365",
            "-subj",
            f"/CN={hookenv.service_name()}.{model}.svc",
            "-nodes",
        ],
        check=True,
    )

    ca_bundle = b64encode(Path('cert.pem').read_bytes()).decode('utf-8')

    layer.caas_base.pod_spec_set(
        {
            'version': 2,
            'serviceAccount': {
                'global': True,
                'rules': [
                    {
                        'apiGroups': ['kubeflow.org'],
                        'resources': ['poddefaults'],
                        'verbs': ['get', 'list', 'watch', 'update', 'create', 'patch', 'delete'],
                    },
                ],
            },
            'containers': [
                {
                    'name': 'admission-webhook',
                    'imageDetails': {
                        'imagePath': image_info.registry_path,
                        'username': image_info.username,
                        'password': image_info.password,
                    },
                    'ports': [{'name': 'webhook', 'containerPort': 443}],
                    'files': [
                        {
                            'name': 'certs',
                            'mountPath': '/etc/webhook/certs',
                            'files': {
                                'cert.pem': Path('cert.pem').read_text(),
                                'key.pem': Path('key.pem').read_text(),
                            },
                        }
                    ],
                }
            ],
        },
        k8s_resources={
            'kubernetesResources': {
                'customResourceDefinitions': {
                    crd['metadata']['name']: crd['spec']
                    for crd in yaml.safe_load_all(Path("files/crds.yaml").read_text())
                },
                'mutatingWebhookConfigurations': {
                    'admission-webhook': [
                        {
                            'name': 'admission-webhook.kubeflow.org',
                            'failurePolicy': 'Fail',
                            'clientConfig': {
                                'caBundle': ca_bundle,
                                'service': {
                                    'name': hookenv.service_name(),
                                    'namespace': model,
                                    'path': '/apply-poddefault',
                                },
                            },
                            "objectSelector": {
                                "matchExpressions": [
                                    {
                                        "key": "juju-app",
                                        "operator": "NotIn",
                                        "values": ["admission-webhook"],
                                    },
                                    {
                                        "key": "juju-operator",
                                        "operator": "NotIn",
                                        "values": ["admission-webhook"],
                                    }
                                ]
                            },
                            'rules': [
                                {
                                    'apiGroups': [''],
                                    'apiVersions': ['v1'],
                                    'operations': ['CREATE'],
                                    'resources': ['pods'],
                                }
                            ],
                        },
                    ]
                },
            }
        },
    )

    layer.status.maintenance('creating container')
    set_flag('charm.started')
