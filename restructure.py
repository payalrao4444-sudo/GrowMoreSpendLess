import os
import shutil

base_dir = r"c:\Users\DELL\OneDrive\Desktop\claud\dataset\disease"
veg_folders = ['tomato', 'potato', 'cauliflower', 'eggplant']

for veg in veg_folders:
    veg_path = os.path.join(base_dir, veg)
    if os.path.isdir(veg_path):
        for item in os.listdir(veg_path):
            item_path = os.path.join(veg_path, item)
            if os.path.isdir(item_path):
                # Ensure the new name indicates both vegetable and disease
                # e.g., if item is "Black Rot", make it "Cauliflower_Black_Rot"
                new_name = item.replace(' ', '_')
                
                # if the name already contains the veg name, leave it (like Tomato_Early_blight)
                if veg.lower() not in new_name.lower():
                    # special case for Augmented folders
                    if "augmented" in new_name.lower():
                        new_name = new_name.replace("Augmented_", "").replace("Augmented", "").strip("_ ")
                        if veg.lower() not in new_name.lower():
                            new_name = f"{veg.capitalize()}_{new_name}"
                    else:
                        new_name = f"{veg.capitalize()}_{new_name}"
                
                dest_path = os.path.join(base_dir, new_name)
                
                if os.path.exists(dest_path):
                    # Handle merge or overwrite
                    for f in os.listdir(item_path):
                        shutil.move(os.path.join(item_path, f), os.path.join(dest_path, f))
                    os.rmdir(item_path)
                else:
                    shutil.move(item_path, dest_path)
        
        # After moving all subdirectories, remove the vegetable directory
        if not os.listdir(veg_path):
            os.rmdir(veg_path)
            
print("Dataset restructuring complete.")
