import yaml


def convert(charm: str, name: str, namespace: str) -> list:
    """Converts charm.yaml to list of resources to send to Kubernetes API."""

    charm = yaml.safe_load(charm)
    resources = []

    if charm['permissions']:
        resources += [
            {'apiVersion': 'v1', 'kind': 'ServiceAccount', 'metadata': {'name': name}},
            {
                'apiVersion': 'rbac.authorization.k8s.io/v1',
                'kind': 'ClusterRole',
                'metadata': {'name': name},
                'rules': charm['permissions']['rules'],
            },
            {
                'apiVersion': 'rbac.authorization.k8s.io/v1',
                'kind': 'ClusterRoleBinding',
                'metadata': {'name': name},
                'roleRef': {
                    'apiGroup': 'rbac.authorization.k8s.io',
                    'kind': 'ClusterRole',
                    'name': name,
                },
                'subjects': [{'kind': 'ServiceAccount', 'name': name, 'namespace': namespace}],
            },
        ]

    if charm['containers']:
        ports = [
            {
                'name': port['name'],
                'port': port['expose'],
                'targetPort': port.get('bind', port['expose']),
            }
            for container in charm['containers']
            for port in container.get('ports')
        ]
        resources += [
            {
                'apiVersion': 'apps/v1',
                'kind': 'Deployment',
                'metadata': {'name': name},
                'spec': {
                    'selector': {'matchLabels': {'app': name}},
                    'template': {
                        'metadata': {'labels': {'app': name, 'juju-app': name}},
                        'spec': {
                            'containers': [
                                {
                                    'name': container['name'],
                                    'image': container['image'],
                                    'args': container.get('args', []),
                                    'volumeMounts': [
                                        {
                                            'mountPath': '/etc/webhook/certs',
                                            'name': 'webhook-cert',
                                            'readOnly': True,
                                        }
                                    ],
                                }
                                for container in charm['containers']
                            ],
                            'serviceAccountName': name,
                            'volumes': [
                                {
                                    'name': 'webhook-cert',
                                    'secret': {'secretName': 'admission-webhook-cert-tls'},
                                }
                            ],
                        },
                    },
                },
            },
            {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {'name': name},
                'spec': {'ports': ports},
            },
        ]

    return resources
