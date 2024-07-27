import string
import random

# Generate a random string of specified length
def random_string(length = 10):
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string