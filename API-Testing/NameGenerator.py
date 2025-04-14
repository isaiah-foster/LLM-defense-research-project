# Description: Generates random names from a list of first and last name txt files into a list of 5 random names.
import random

# Read first and last names from txt files to new lists
with open("API-Testing/Names/FirstNames.txt", "r", encoding="utf-8") as f:
    first_names = [line.strip() for line in f if line.strip()]

with open("API-Testing/Names/LastNames.txt", "r", encoding="utf-8") as f:
    last_names = [line.strip() for line in f if line.strip()]

# Generate {n} random name pairs
def generate_random_names(n):
    random_names = [f"{random.choice(first_names)} {random.choice(last_names)}" for _ in range(n)]
    return random_names