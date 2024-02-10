import os

def delete_files_after_200(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        file_count = len(filenames)
        if file_count > 500:
            files_to_delete = filenames[200:]
            for file_to_delete in files_to_delete:
                file_path = os.path.join(dirpath, file_to_delete)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# Example usage:
root_directory = "data/asl_train/asl_alphabet_train/"
delete_files_after_200(root_directory)
