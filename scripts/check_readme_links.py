import os
import re
import sys

with open("README.md", encoding="utf-8") as f:
    content = f.read()

md_links = re.findall(r"\[([^\]]*)\]\(([^)]+)\)", content)
html_links = re.findall(r'(?:href|src)="([^"]+)"', content)

all_targets = [link_tuple[1] for link_tuple in md_links] + html_links
print(f"Total links/assets found: {len(all_targets)}")

missing = []
for link in all_targets:
    if link.startswith("http") or link.startswith("#") or link.startswith("mailto:"):
        continue
    clean = link.split("?")[0].split("#")[0]
    if not clean:
        continue
    clean = clean.replace("/", os.sep)
    if not os.path.exists(clean):
        missing.append((link, clean))

if missing:
    print("FAILED! Missing local targets:")
    for orig, clean in missing:
        print(f"  {orig} -> {clean}")
    sys.exit(1)
else:
    print("SUCCESS: All local file and asset paths resolve cleanly!")
