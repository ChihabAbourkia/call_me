import json
from .json_validator import Prompts ,Functions

def prompts_loader(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
        return [Prompts(**item) for item in data]
    except FileNotFoundError as e:
        print(f"ERROR : {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR : {e}")
        exit(1)


def function_loader(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
        return [Functions(**item) for item in data]
    except FileNotFoundError as e:
        print(f"ERROR : {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR : {e}")
        exit(1)
