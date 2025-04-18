import random

first_names = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer",
    "Michael", "Linda", "William", "Elizabeth", "David", "Barbara"
]

last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez"
]

def generate_name_list(n):
    return [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n)]