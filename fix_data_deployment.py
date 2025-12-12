import shutil
import os

SOURCE_DIR = r"c:\Coding\DVProject\dashboard_data"
DEST_DIR = r"c:\Coding\DVProject\dv-dashboard\public\data"

def fix_data():
    print(f"Checking Source: {SOURCE_DIR}")
    if not os.path.exists(SOURCE_DIR):
        print("ERROR: Source directory missing!")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.json')]
    print(f"Found files: {files}")

    print(f"Ensuring Destination: {DEST_DIR}")
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print("Created destination directory.")
    
    for f in files:
        src = os.path.join(SOURCE_DIR, f)
        dst = os.path.join(DEST_DIR, f)
        shutil.copy2(src, dst)
        print(f"Copied {f} to {dst}")

    print("Verification:")
    final_files = os.listdir(DEST_DIR)
    print(f"Files in public/data: {final_files}")

if __name__ == "__main__":
    fix_data()
