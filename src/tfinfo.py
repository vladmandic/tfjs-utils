import tensorflow as tf

print('sysconfig:', tf.sysconfig.get_build_info())

for device in tf.config.list_physical_devices():
  print('gpu device:', device, tf.config.experimental.get_device_details(device))

for device in tf.config.list_logical_devices():
  print('logical device:', device)
