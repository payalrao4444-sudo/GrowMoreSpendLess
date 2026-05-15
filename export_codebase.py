import os
import datetime

# Define the root directory of the project
ROOT_DIR = r"c:\Users\DELL\OneDrive\Desktop\claud"

# The output destination
OUTPUT_FILE = os.path.join(ROOT_DIR, "Complete_Project_Source_Code.txt")

# Files we definitely want to include as the "Full Working Source Code"
FILES_TO_INCLUDE = [
    "app.py",
    "requirements.txt",
    "README.md",
    "train_model.py",
    "plant_disease.py",
    "complete_classifier.py",
    "templates/dashboard.html",
    "templates/login.html",
    "templates/predict.html"
]

def generate_tree(startpath):
    tree_str = "PROJECT STRUCTURE:\nproject_name/\n"
    for current_file in FILES_TO_INCLUDE:
        # Simple tree formatting for the explicitly included files
        parts = current_file.split('/')
        if len(parts) == 1:
            tree_str += f"├── {parts[0]}\n"
        elif current_file == "templates/dashboard.html": # manually formatting the tree layout
            tree_str += "├── templates/\n"
            tree_str += f"│   ├── {parts[1]}\n"
        else:
            tree_str += f"│   ├── {parts[1]}\n"
    tree_str += "├── static/\n└── database/\n\n"
    return tree_str

def build_consolidated_file():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # Write Header
        outfile.write("=========================================================================\n")
        outfile.write("             AI SMART KITCHEN GARDENING SYSTEM - SOURCE CODE             \n")
        outfile.write(f"             Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}             \n")
        outfile.write("=========================================================================\n\n")
        
        # Write Project Structure
        outfile.write(generate_tree(ROOT_DIR))
        outfile.write("=========================================================================\n\n")

        # Write each file
        for rel_path in FILES_TO_INCLUDE:
            abs_path = os.path.join(ROOT_DIR, rel_path.replace("/", "\\"))
            if os.path.exists(abs_path):
                outfile.write("===================================\n")
                outfile.write(f"FILE: {rel_path}\n")
                outfile.write("===================================\n")
                
                try:
                    with open(abs_path, "r", encoding="utf-8") as infile:
                        content = infile.read()
                        
                        # (Optional) Basic cleanup before writing to output
                        # We strip excess trailing whitespace to keep the document clean
                        cleaned_lines = [line.rstrip() for line in content.split('\n')]
                        outfile.write('\n'.join(cleaned_lines))
                        
                except Exception as e:
                    outfile.write(f"Error reading file: {e}\n")
                
                outfile.write("\n\n\n")
            else:
                outfile.write("===================================\n")
                outfile.write(f"FILE: {rel_path} (NOT FOUND)\n")
                outfile.write("===================================\n\n\n")

if __name__ == "__main__":
    print(f"Aggregating full source code into a single document...")
    build_consolidated_file()
    print(f"SUCCESS: Source code successfully exported to:\n{OUTPUT_FILE}")
