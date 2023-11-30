import sqlite3
import json
import scrapy
import os
import urllib.parse as url
from scrapy.crawler import CrawlerProcess
from scrapy.shell import inspect_response


class AllRecipesSpider(scrapy.Spider):
    name = "allrecipes"

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        # enabled_ext = {"scrapy.extensions.closespider.CloseSpider": 100}
        # settings.set("EXTENSIONS", {})
        settings.set("CLOSESPIDER_ITEMCOUNT", 100, priority="spider")
        settings.set(
            "FEEDS", {"allrecipes.jl": {"format": "jsonlines", "overwrite": True}}
        )

    def start_requests(self):
        url = "https://www.allrecipes.com/recipes-a-z-6735880"
        yield scrapy.Request(url, self.parse_a_z)

    def parse_a_z(self, response):
        # Capture all category items in A-Z list
        for category in response.xpath(
            "//li[contains(@id,'link-list__item_1-0')]/a[contains(@href,'recipes')]/@href"
        ):
            yield scrapy.Request(category.get(), self.parse_categories)

    def parse_categories(self, response):
        url_path_components = url.urlsplit(response.url)[2].strip("/").split("/")
        main_category = url_path_components[-1]
        print(main_category)
        # Capture all subcategory links
        for category in response.xpath(
            f"//a[contains(@href, '{main_category}')]/span[@class='link__wrapper']/parent::*/@href"
        ).getall():
            yield scrapy.Request(category, self.parse_categories)

        # Capture all recipes on this page
        recipes_json = json.loads(
            response.xpath("//script[@id='allrecipes-schema_1-0']/text()").get()
        )

        # Urls found in json
        for recipe_url in recipes_json[0]["itemListElement"]:
            # Get type of webpage linked (recipe, gallery, article)
            # Only recipe pages have the data we want
            if url.urlparse(recipe_url["url"]).path.split("/")[1] == "recipe":
                yield scrapy.Request(recipe_url["url"], self.parse_recipe)

    def parse_recipe(self, response):
        recipe = json.loads(
            response.xpath("//script[@id='allrecipes-schema_1-0']/text()").get()
        )[0]

        # Recipe information found in json
        ingredients = recipe["recipeIngredient"]
        instructions = recipe["recipeInstructions"]

        def get_possibly_missing_key(data, key, default=0):
            if key in data.keys():
                val = data[key]
            else:
                val = default
            return val

        prep_time = get_possibly_missing_key(recipe, "prepTime", default=0)
        cook_time = get_possibly_missing_key(recipe, "cookTime", default=0)
        total_time = get_possibly_missing_key(recipe, "totalTime", default=0)
        yield {
            "recipe_name": recipe["name"],
            "time": {
                "prep": prep_time,
                "cook": cook_time,
                "total": total_time,
            },
            "ingredients": ingredients,
            "instructions": instructions,
            "body": json.dumps(recipe),
        }
