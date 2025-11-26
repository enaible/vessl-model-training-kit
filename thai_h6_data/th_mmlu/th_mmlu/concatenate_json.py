import json
import os
from pathlib import Path
from typing import Any, List

def gather_and_concatenate_json(directory: str = ".", output_file: str = "th_mmlu.json") -> None:
    """
    Gather all JSON files matching pattern th_mmlu__{subject}__test.json
    in the specified directory and concatenate them into a single JSON file.

    Args:
        directory: Directory to search for JSON files (default: current directory)
        output_file: Name of the output JSON file (default: th_mmlu.json)
    """
    # Find all matching JSON files
    matching_files = sorted(Path(directory).glob("th_mmlu__*__test.json"))

    if not matching_files:
        print(f"No matching JSON files found in {directory}")
        return

    print(f"Found {len(matching_files)} matching file(s):")
    for file in matching_files:
        print(f"  - {file.name}")

    concatenated_data: List[Any] = []

    # Read and concatenate all JSON files
    for file in matching_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # If the data is a list, extend; if it's a dict, append
                if isinstance(data, list):
                    concatenated_data.extend(data)
                else:
                    concatenated_data.append(data)

                print(f"✓ Loaded {file.name}")
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing {file.name}: {e}")
        except Exception as e:
            print(f"✗ Error reading {file.name}: {e}")

    # Write concatenated data to output file
    try:
        output_path = Path(directory) / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(concatenated_data, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Successfully created {output_file} with {len(concatenated_data)} items")
    except Exception as e:
        print(f"✗ Error writing output file: {e}")


if __name__ == "__main__":
    # Example usage - you can modify the directory path if needed
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = sys.argv[2] if len(sys.argv) > 2 else "th_mmlu.json"

    gather_and_concatenate_json(directory, output_file)