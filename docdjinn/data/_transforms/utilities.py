import hashlib
import json

def generate_transform_hash(kwargs_dict):
    """Generate a unique hash for transform kwargs."""
    # Sort the dictionary to ensure consistent hashing
    sorted_kwargs = json.dumps(kwargs_dict, sort_keys=True, default=str)
    return hashlib.md5(sorted_kwargs.encode()).hexdigest()[:8]
