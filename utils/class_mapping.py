import os
def class_mapping(base_directory):
    """
    Create a mapping between class indices and sub-directory names.

    Parameters:
    - base_directory (str): Path to the directory containing sub-directories (classes).

    Returns:
    - class_mapping (dict): Mapping between class indices and sub-directory names.
    """
    # Get a list of sub-directory names
    subdirectories = sorted(next(os.walk(base_directory))[1])

    # Create a mapping between class indices and sub-directory names
    class_mapping = {i: subdir for i, subdir in enumerate(subdirectories)}

    return class_mapping


# Example usage:
base_directory_path = '/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset/train'
mapping = class_mapping(base_directory_path)

# Print the mapping
print("Class Mapping:")
for class_id, class_name in mapping.items():
    print(f"Class {class_id}: {class_name}")
