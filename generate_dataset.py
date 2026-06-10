from __future__ import annotations

import csv
from pathlib import Path
import random

OUTPUT_PATH = Path("data/names_gender.csv")

MALE_NAMES = [
    "Venkata", "Srinivasarao", "Sathyanarayan", "Ramesh", "Nagaraju", "Apparao", "Suresh", "Ramu", "Satyanarayana", "Srinivas",
    "Prasad", "Murali", "Subramanyam", "Koteswararao", "Venkatesh", "Rama", "Ramana", "Shiva", "Krishna", "Mahesh",
    "Rajesh", "Ashok", "Anil", "Hari", "Naveen", "Praveen", "Rakesh", "Sandeep", "Srikanth", "Vijay",
    "Kumar", "Pavan", "Balaji", "Nagarjuna", "Gopal", "Ganesh", "Mohan", "Narayana", "Ravindra", "Chandra",
    "Sridhar", "Babu", "Baburao", "Raj", "Rajendra", "Anand", "Naresh", "Venkat", "Vasu", "Lokesh",
    "Venkatasubbaiah", "Satyanand", "Ranganath", "Ramanatha", "Ramachandra", "Sitharam", "Sudhakar", "Prabhakar", "Bhaskar", "Madhusudhan",
    "Gangadhar", "Suryanarayana", "Venkateswararao", "Srinadh", "Nagaraja", "Subbarao", "Koteswara", "Eswar", "Siva", "Aditya",
    "Karthik", "Siddharth", "Vikram", "Abhishek", "Rahul", "Tarun", "Kiran", "Bharath", "Sai", "Satish",
    "Kalyan", "Giri", "Govind", "Jagadeesh", "Raghu", "Srinivasu", "Pardhasaradhi", "Dinesh", "Naidu", "Chowdary",
    "Balaram", "Raghavendra", "Venkanna", "Mallikarjuna", "Dharma", "Kondalarao", "Chiranjeevi", "Balakrishna", "Narayanarao", "Narasimha",
    "Manikanta",
]

FEMALE_NAMES = [
    "Lakshmi", "Satyavathi", "Nagamani", "Ramanamma", "Durga", "Vijaya", "Varalakshmi", "Padma", "Lakshmidevi", "Parvathi",
    "Narayanamma", "Ramalakshmi", "Jayalakshmi", "Aadilakshmi", "Vijailaxmi", "Sujatha", "Sreedevi", "Sarojini", "Anuradha", "Rangamma",
    "Kalyani", "Prabhavathi", "Rajyalakshmi", "Bhavani", "Madhavi", "Sailaja", "Sowjanya", "Lavanya", "Sravani", "Anusha",
    "Revathi", "Mounika", "Anjali", "Sirisha", "Suneetha", "Sridevi", "Geetha", "Radha", "Savitri", "Kavitha",
    "Kalpana", "Nirmala", "Lalitha", "Usha", "Manjula", "Bhargavi", "Bhagyamma", "Padmavathi", "Saraswathi", "Swarna",
    "Sita", "Ganga", "Lakshmamma", "Venkatalakshmi", "Adilakshmi", "Durgamma", "Rajeswari", "Gowri", "Savitramma", "Subbamma",
    "Kanakadurga", "Ananthalakshmi", "Annapurna", "Lalithadevi", "Syamala", "Sunitha", "Geetanjali", "Priyanka", "Deepika", "Keerthi",
    "Sandhya", "Swapna", "Divya", "Jyothi", "Tejaswini", "Sindhu", "Harika", "Divyasree", "Siri", "Prasanna",
    "Asha", "Amani", "Hemalatha", "Leela", "Malathi", "Pushpa", "Renuka", "Sujani", "Vani", "Vasundhara",
    "Padmaja", "Durgadevi", "Satyabhama", "Arundhati", "Indravathi", "Chandrakala", "Sneha", "Meenakshi", "Yamini", "Shanti",
]


def build_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    for name in MALE_NAMES:
        rows.append((name, "male"))
        rows.append((name.lower(), "male"))

    for name in FEMALE_NAMES:
        rows.append((name, "female"))
        rows.append((name.lower(), "female"))

    # Add a few noisy variants so the classifier sees simple formatting differences.
    variants = []
    for name, gender in rows:
        variants.append((name.strip(), gender))
        variants.append((name.title(), gender))

    rows.extend(variants)
    random.Random(42).shuffle(rows)
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["name", "gender"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
