import os

def find_string_in_files(directory, query_list):
    print(f"Searching for queries in {directory}...")
    for root, dirs, files in os.walk(directory):
        if "data_new" in root or ".git" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    for idx, line in enumerate(lines):
                        for query in query_list:
                            if query in line:
                                print(f"{path}:{idx+1}: {line.strip()}")
                except Exception as e:
                    pass

if __name__ == "__main__":
    find_string_in_files(r"D:\Bioview\My_RF_work_v1", ["7.91", "t_start =", "t_start=", "axvline(7.9", "axvline(8."])
