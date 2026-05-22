import json
from pathlib import Path

data = json.loads(Path("raw_article.json").read_text(encoding="utf-8"))

print(data["article_id"])
print(data["title"])
print(data["body"][:60])

output = {"message": "こんにちは", "items": ["A", "B", "C"]}

Path("output.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
)
