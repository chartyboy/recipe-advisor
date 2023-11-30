import json
import locale

print(locale.getdefaultlocale())
data = []
with open("./datasets/raw/recipes.jl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open("./datasets/raw/sample.jl", "w", encoding="utf-8") as f:
    for recipe in data[:100]:
        # del recipe["body"]
        json.dump(recipe, f)
        f.write("\n")
