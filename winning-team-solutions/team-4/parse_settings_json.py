import json
import argparse


def get_raw_data_dir(key):
    with open("SETTINGS.json", "r") as f:
        settings = json.load(f)
        return settings.get(key, "")  # Provide a default value as fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get value from settings.json")
    parser.add_argument("key", type=str, help="Key")
    args = parser.parse_args()

    print(get_raw_data_dir(args.key))
