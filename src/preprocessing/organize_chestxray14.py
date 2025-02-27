import os
import shutil


def organize_images(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Only process files with .png extension
        if filename.endswith(".png"):
            # Split the filename into the prefix and suffix
            prefix, suffix = filename.split("_")

            # Create the subdirectory path
            subdirectory = os.path.join(directory, prefix)

            # Create the subdirectory if it doesn't exist
            os.makedirs(subdirectory, exist_ok=True)

            # Create the new file name (suffix only)
            new_filename = suffix

            # Define the source and destination paths
            src_path = os.path.join(directory, filename)
            dst_path = os.path.join(subdirectory, new_filename)

            # Move the file
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {dst_path}")


if __name__ == "__main__":
    path = "/path/to/ChestX-ray14/images"
    organize_images(directory=path)
