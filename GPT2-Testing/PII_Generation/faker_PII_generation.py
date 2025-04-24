from faker import Faker
import random

# Initialize Faker and seed for reproducibility
fake = Faker()
Faker.seed(42)
random.seed(42)

# Set how many entries you want
<<<<<<< HEAD
num_entries = 50
=======
num_entries = 25
>>>>>>> 1d7bd6dd21b6b61b8f04286d519620c52e250976

# File to write to
output_file = "GPT2-Testing/PII_Generation/synthetic_pii.txt"

# Open file for writing
with open(output_file, "w") as f:
    for _ in range(num_entries):
        name = fake.name()
        address = fake.address().replace("\n", ", ")
        email = fake.email()
        phone = fake.phone_number()
        sentence = fake.sentence()

        line = (
            f"{sentence} Contact {name} via email at {email}, "
            f"phone number {phone}, or visit them at {address}.\n"
        )

        f.write(line)

print(f"Saved {num_entries} entries to {output_file}")
