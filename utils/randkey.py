_ = """
This file is used to generate a random string of specified length as a mark for highlighting part.
"""

import string
import random

# Generate a random string of specified length
def random_string(length = 10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string