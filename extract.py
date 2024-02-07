import subprocess

# Define the path to the WARC file in the Downloads directory
warc_file_path = r"C:/Users/Aditi Agarwal/Downloads/LargeData.warc"

# Define the path to the Pythia folder in your main directory
output_directory = r"C:/Users/Aditi Agarwal/Desktop/Pythia"

# Command to run the warc-extractor tool
command = ["warc-extractor", warc_file_path, output_directory]

try:
    # Run the command using subprocess
    result = subprocess.run(command, check=True)
    if result.returncode == 0:
        print("Extraction completed successfully!")
    else:
        print("Extraction failed with return code:", result.returncode)
except subprocess.CalledProcessError as e:
    print("Error:", e)

