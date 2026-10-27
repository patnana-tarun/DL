"""EdgeVision: camera / stream -> person detector -> tracker -> crowd analytics -> VLM, on one device.

The same code runs on a desktop GPU and on an NVIDIA Jetson; only the config file changes.

    python -m edgevision --config configs/desktop.yaml
"""
__version__ = "1.0.0"
