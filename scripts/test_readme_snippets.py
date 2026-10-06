import re
import sys

with open("README.md", encoding="utf-8") as f:
    content = f.read()

python_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)
print(f"Found {len(python_blocks)} python snippets in README.md")

scope = {}
for idx, block in enumerate(python_blocks):
    print(f"\n--- Testing Python Block {idx + 1} ---")
    code = block.strip()
    try:
        exec(code, scope)
        print(f"Block {idx + 1} PASSED cleanly!")
    except Exception as e:
        print(f"Block {idx + 1} FAILED with error: {e}")
        sys.exit(1)

print("\nAll README Python code snippets executed successfully!")
