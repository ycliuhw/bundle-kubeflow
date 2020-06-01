#!/usr/bin/env python3

import os
from base64 import b64encode
from pathlib import Path
from subprocess import run

import yaml
from ops.charm import CharmBase
from ops.main import main
from ops.model import ActiveStatus, MaintenanceStatus

from charmhelpers.core import hookenv
from oci_image import OCIImageResource, OCIImageResourceError


class AdmissionWebhookCharm(CharmBase):
    def __init__(self, framework, key):
        super().__init__(framework, key)
        self.image = OCIImageResource(self, 'oci-image')
        self.framework.observe(self.on.install, self.set_pod_spec)
        self.framework.observe(self.on.upgrade_charm, self.set_pod_spec)

    def set_pod_spec(self, event):
        if not self.model.unit.is_leader():
            print('Not a leader, skipping set_pod_spec')
            self.model.unit.status = ActiveStatus()
            return

        self.model.unit.status = MaintenanceStatus('Setting pod spec')

        try:
            image_details = self.image.fetch()
        except OCIImageResourceError as e:
            self.model.unit.status = e.status
            return

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

        self.model.pod.set_spec(
            {
                'version': 3,
                'serviceAccount': {
                    'roles': [
                        {
                            'global': True,
                            'rules': [
                                {
                                    'apiGroups': ['kubeflow.org'],
                                    'resources': ['poddefaults'],
                                    'verbs': [
                                        'get',
                                        'list',
                                        'watch',
                                        'update',
                                        'create',
                                        'patch',
                                        'delete',
                                    ],
                                },
                            ],
                        }
                    ],
                },
                'containers': [
                    {
                        'name': 'admission-webhook',
                        'imageDetails': image_details,
                        'ports': [{'name': 'webhook', 'containerPort': 443}],
                        'volumeConfig': [
                            {
                                'name': 'certs',
                                'mountPath': '/etc/webhook/certs',
                                'files': [
                                    {'path': 'cert.pem', 'content': Path('cert.pem').read_text()},
                                    {'path': 'key.pem', 'content': Path('key.pem').read_text()},
                                ],
                            }
                        ],
                    }
                ],
            },
            k8s_resources={
                'kubernetesResources': {
                    'customResourceDefinitions': {
                        crd['metadata']['name']: crd['spec']
                        for crd in yaml.safe_load_all(Path("src/crds.yaml").read_text())
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
                                        },
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

        self.model.unit.status = ActiveStatus()


if __name__ == '__main__':
    main(AdmissionWebhookCharm)
