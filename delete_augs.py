import os

root_dir = r"C:\Users\PC\Documents\Image-forgery\recodai-luc-scientific-image-forgery-detection"  # replace with your root folder

for folder, subfolders, files in os.walk(root_dir):
    for file in files:
        if "aug" in file:
            file_path = os.path.join(folder, file)
            print(f"Deleting: {file_path}")
            os.remove(file_path)
