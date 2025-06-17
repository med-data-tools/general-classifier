import json
import os

def save_data_to_json(data, filename):
    """Saves data to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data saved to {filename}")
        return True
    except IOError as e:
        print(f"Error saving data to {filename}: {e}")
        return False

def load_data_from_json(filename):
    """Loads data from a JSON file."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except IOError as e:
        print(f"Error loading data from {filename}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filename}: {e}")
        return None
import os # This import might be duplicated if it's also at the top, but it's fine.
