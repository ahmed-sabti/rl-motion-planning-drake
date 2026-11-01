import math
import numpy as np

def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value
