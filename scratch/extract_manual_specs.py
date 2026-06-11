import fitz
import re

doc = fitz.open('D:/Stethoscope/ES-Patch MANUAL.pdf')
print("Total Pages:", len(doc))

out_lines = []
out_lines.append("# Electronic Stethoscope Patch (ES-Patch) User's Manual - Technical Specifications\n")
out_lines.append(f"**File**: `D:\\Stethoscope\\ES-Patch MANUAL.pdf`  ")
out_lines.append(f"**Pages**: {len(doc)}  ")
out_lines.append(f"**Version**: V1-2  \n")

# Keywords to find
keywords = ["specification", "technical", "mode", "frequency", "sampling", "filter", "performance", "range", "es-patch", "sound", "sensor"]

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()
    
    # Check if page has any relevant keyword
    has_keyword = any(kw in text.lower() for kw in keywords)
    if has_keyword:
        out_lines.append(f"\n## PAGE {page_num + 1}\n")
        out_lines.append(text)

with open('scratch/es_patch_specifications.md', 'w', encoding='utf-8') as f:
    f.write("\n".join(out_lines))

print("Extraction complete. Saved to scratch/es_patch_specifications.md")
