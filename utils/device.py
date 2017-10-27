from tensorflow.python.client import device_lib


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def normalize_device_name(name):
    name = name.lower().replace('device:', '')
    return name
