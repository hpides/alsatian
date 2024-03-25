import random
import string

alphabet = string.ascii_lowercase + string.digits

def random_short_id():
    return ''.join(random.choices(alphabet, k=8))