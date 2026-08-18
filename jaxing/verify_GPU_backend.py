import jax

# List all available devices
print("Available devices:", jax.devices())

# Explicitly check the default backend device
current_device = jax.devices()[0]
print(f"Using device: {current_device.device_kind} ({current_device.platform.upper()})")
