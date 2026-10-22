import os
import re
import sys

with open("README.md", encoding="utf-8") as f:
    content = f.read()

md_links = re.findall(r"\[([^\]]*)\]\(([^)]+)\)", content)
html_links = re.findall(r'(?:href|src)="([^"]+)"', content)

all_targets = [link_tuple[1] for link_tuple in md_links] + html_links
print(f"Total links/assets found: {len(all_targets)}")

# Collect explicit IDs and heading slugs for anchor validation
headings = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
valid_anchors = set()
for _, h in headings:
    # Standard slug (collapsed spaces)
    slug = re.sub(r"[^\w\s-]", "", h.lower()).strip()
    slug_collapsed = re.sub(r"[\s]+", "-", slug)
    valid_anchors.add(slug_collapsed)
    # GitHub slug (uncollapsed spaces, e.g. ' & ' -> '  ' -> '--')
    slug_github = re.sub(r"[^\w\s-]", "", h.lower()).strip().replace(" ", "-")
    valid_anchors.add(slug_github)

explicit_ids = set(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', content))
valid_anchors.update(explicit_ids)

missing = []
for link in all_targets:
    if link.startswith("http") or link.startswith("mailto:"):
        continue
    if link.startswith("#"):
        anchor = link[1:]
        if anchor not in valid_anchors:
            missing.append((link, f"anchor #{anchor} not found in headings or explicit IDs"))
        continue
    clean = link.split("?")[0].split("#")[0]
    if not clean:
        continue
    clean = clean.replace("/", os.sep)
    if not os.path.exists(clean):
        missing.append((link, clean))

if missing:
    print("FAILED! Missing local targets or anchors:")
    for orig, clean in missing:
        print(f"  {orig} -> {clean}")
    sys.exit(1)
else:
    print("SUCCESS: All local file, asset paths, and internal anchors resolve cleanly!")
