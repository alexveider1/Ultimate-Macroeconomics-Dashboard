"""Shared inference logic baked into the Triton image (on ``PYTHONPATH``).

Every Triton python-backend ``model.py`` is a thin shell that decodes a JSON
request tensor, calls into this package, and encodes a JSON response tensor.
Keeping the real fit/predict maths here (instead of in each ``model.py``) means
the numeric behaviour is unit-testable on CPU without a running Triton server.
"""
