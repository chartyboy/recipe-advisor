import scrapy
import json
import re


def parse_json_recipe(response):
    try:
        recipe = json.loads(
            response.xpath("//script[@type='application/ld+json']/text()").get()
        )
        # First type of JSON schema used for aggregate websites
        # List of dicts of website elements, where the first element contains
        # the info needed
        if isinstance(recipe, list):
            recipe = recipe[0]

        # Second type of JSON schema, dict with @graph key corresponding
        # to list of dicts for site elements
        elif isinstance(recipe, dict):
            if "@graph" in recipe.keys():
                for content_dict in recipe["@graph"]:
                    if content_dict["@type"].lower() == "recipe":
                        recipe = content_dict
                        break
        else:
            raise TypeError("")
    except:
        return None

    def get_possibly_missing_key(data, key, default=0):
        if key in data.keys():
            val = data[key]
        else:
            val = default
        return val

    # Recipe information found in json
    keys_and_defaults = {
        "name": "",
        "recipeIngredient": [],
        "recipeInstructions": [],
        "prepTime": 0,
        "cookTime": 0,
        "totalTime": 0,
        "recipeYield": 0,
    }

    recipe_info = dict()
    for key, default in keys_and_defaults.items():
        recipe_info[key] = get_possibly_missing_key(recipe, key, default=default)

    return {
        "recipe_name": recipe_info["name"],
        "yields": recipe_info["recipeYield"],
        "time": {
            "prep": recipe_info["prepTime"],
            "cook": recipe_info["cookTime"],
            "total": recipe_info["totalTime"],
        },
        "ingredients": recipe_info["recipeIngredient"],
        "instructions": recipe_info["recipeInstructions"],
        "body": json.dumps(recipe),
        "source_url": response.url,
    }


def replace_newlines(string: str, repl: str) -> str:
    return re.sub(r"(\n\s+){1,}", repl, string)
