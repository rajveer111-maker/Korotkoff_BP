import os
import re

def search():
    pattern = re.compile(r'butter.*band', re.IGNORECASE)
    for root, dirs, files in os.walk('.'):
        if 'bioview_env' in root or '.git' in root or '.system_generated' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if pattern.search(line) or 'sos_koro' in line or 'sos_k' in line:
                                print(f"{filepath}:{i}: {line.strip()}")
                except Exception as e:
                    pass

if __name__ == '__main__':
    search()
