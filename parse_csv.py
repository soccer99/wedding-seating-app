import sys
import csv
import json


DEBUG = False


def open_csv(filepath):
    with open(filepath, "r") as f:
        return f.read()


def parse_guests(csv_data):
    # Split the data into lines and extract headers
    lines = csv_data.strip().split("\n")
    headers = lines[0].split(",")

    # Initialize the guest list and ID counter
    guests = []
    next_id = 1
    id_mapping = {}  # To keep track of guests and their IDs
    # Process each line of data
    for line in lines[1:]:  # Skip the header line
        values = line.split(",")
        data = dict(zip(headers, values))

        # Extract main guest information
        guest_name = f"{data['first_name']} {data['last_name']}".strip()

        relation = data["relationship_to_couple"]
        guest_age = data["age_range"]

        # Create guest entry and record ID
        guest_id = next_id
        next_id += 1
        id_mapping[guest_name] = guest_id

        # Initialize connections list
        connections = []

        # Check for partner
        partner = {}
        partner_name = None
        if data["partner_first_name"]:
            if data["partner_first_name"].lower().strip() != "guest":
                partner_name = (
                    f"{data['partner_first_name']} {data['partner_last_name']}".strip()
                )
            else:
                partner_name = f"{guest_name}'s guest"

            # Create partner entry and record ID
            partner_id = next_id
            next_id += 1
            id_mapping[partner_name] = partner_id

            # Add partner to connections
            connections.append((partner_id, -50))

            # Create partner dictionary and add to guests list
            partner = {
                "id": partner_id,
                "name": partner_name,
                "last_name": data["partner_last_name"].lower(),
                "age": data["partner_age_range"].strip() or guest_age,
                "interests": (
                    data["partner_interests"].lower().strip().split(" ")
                    if data["partner_interests"]
                    else []
                ),
                "relation": f"{relation} partner",
                "connections": set([(guest_id, -50)]),  # Connect back to main guest
            }

        # Check for children
        for i in range(1, 6):  # Check all 5 possible children
            child_first = data.get(f"child_{i}_first_name", "")
            child_last = data.get(f"child_{i}_last_name", "")

            if child_first and child_last:
                child_name = f"{child_first} {child_last}".strip()
                child_id = next_id
                next_id += 1
                id_mapping[child_name] = child_id

                # Add child to connections
                connections.append((child_id, -40))

                # Create child dictionary and add to guests list
                child = {
                    "id": child_id,
                    "name": child_name,
                    "last_name": child_last.lower(),
                    "age": "child",
                    "interests": [],
                    "relation": f"Child of {guest_name}"
                    + (f" and {partner_name}" if partner_name else ""),
                    "connections": [(guest_id, -40)]
                    + ([(partner_id, -40)] if partner_name else []),
                }
                guests.append(child)

        # Add main guest to the list
        guest = {
            "id": guest_id,
            "name": guest_name,
            "last_name": data["last_name"].lower(),
            "age": guest_age,
            "interests": (
                data["interests"].lower().strip().split(" ")
                if data["interests"]
                else []
            ),
            "relation": relation,
            "connections": list(connections),
        }
        guests.append(guest)

        if partner:
            for con in connections:
                if con[0] != partner["id"]:
                    partner["connections"].add(con)
            partner["connections"] = list(partner["connections"])
            guests.append(partner)

    return guests


def create_guest_csv(guest_data):
    guests = sorted(guest_data, key=lambda g: g["id"])
    grid = [[""]]

    for i, guest in enumerate(guests):
        if DEBUG:
            print("guest", guest["name"], guest["id"], guest["connections"])
        row = []
        grid[0].append(guest["name"])

        row.append(guest["name"])
        for e, sub_guest in enumerate(guests):
            if sub_guest["id"] == guest["id"]:
                continue
            if e > i:
                break
            if DEBUG:
                print(
                    "sub_guest",
                    sub_guest["name"],
                    sub_guest["id"],
                    sub_guest["connections"],
                )

            connection_score = 0

            # Prioritize partners
            for con in guest["connections"]:
                if DEBUG:
                    print(con)
                if con[0] == sub_guest["id"]:
                    connection_score += con[1]

            # Higher score if last name matches
            if guest["last_name"] == sub_guest["last_name"]:
                connection_score -= 20

            interests = set(guest["interests"])
            other_interests = set(sub_guest["interests"])

            common_interests = list(interests & other_interests)
            for ci in common_interests:
                connection_score -= 50 if "!" in ci else 10

            if guest["relation"] == sub_guest["relation"]:
                connection_score -= 5

            if guest["age"] == sub_guest["age"]:
                connection_score -= 2

            row.append(min(max(connection_score, -50), 50) or "")
        grid.append(row)

    with open("guest_connections.csv", "w") as write_file:
        writer = csv.writer(write_file)
        writer.writerows(grid)


if __name__ == "__main__":
    print("Running guest parsing")
    csv_data = open_csv(sys.argv[1])
    print("opened source file")
    guests = parse_guests(csv_data)
    print(guests)
    print(len(guests))
    create_guest_csv(guests)
    print("done")
