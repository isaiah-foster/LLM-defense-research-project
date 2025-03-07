# Description: Generates random names from a list of first and last name txt files into a list of 5 random names.
import random

# Read first and last names
with open("Names/FirstNames.txt", "r", encoding="utf-8") as f:
    first_names = [line.strip() for line in f if line.strip()]

with open("Names/LastNames.txt", "r", encoding="utf-8") as f:
    last_names = [line.strip() for line in f if line.strip()]

# Generate 5 random name pairs
def generate_random_names():
    random_names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(5)]
    return random_names