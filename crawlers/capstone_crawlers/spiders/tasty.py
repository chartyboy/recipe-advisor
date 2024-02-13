import sqlite3
import json
import scrapy
import os
import urllib.parse as url
import re
from scrapy.crawler import CrawlerProcess
from scrapy.shell import inspect_response

from capstone_crawlers.spiders.spider_utils import parse_json_recipe


class TastySpider(scrapy.Spider):
    name = "tasty"
    custom_settings = {
        "FEEDS": {"tasty.jl": {"format": "jsonlines", "overwrite": True}},
        "LOG_LEVEL": "INFO",
        # "CLOSESPIDER_PAGECOUNT": 100,
        "DOWNLOAD_DELAY": 0.1,
        "CONCURRENT_REQUESTS": 1,
    }

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        # enabled_ext = {"scrapy.extensions.closespider.CloseSpider": 100}
        # settings.set("EXTENSIONS", {})
        # settings.set("CLOSESPIDER_PAGECOUNT", 100, priority="spider")
        # settings.set("CLOSESPIDER_ITEMCOUNT", 10, priority="spider")
        # settings.set("FEEDS", {"tasty.jl": {"format": "jsonlines", "overwrite": True}})
        # settings.set("LOG_LEVEL", "DEBUG")

    def start_requests(self):
        url = "https://tasty.co/ingredient"
        yield scrapy.Request(url, self.parse_ingredients)

    def parse_ingredients(self, response):
        # Capture all category items from A-Z
        for heading in response.xpath("//li[@class='grouped-list__item']/a/@href"):
            next_page = response.urljoin(heading.get())
            if next_page is not None:
                yield scrapy.Request(next_page, self.parse_categories)

    def parse_categories(self, response):
        # Get number of recipes available for this ingredient
        try:
            n_recipes = int(
                response.xpath("//span[@class='feed-page__heading-count']/text()").get()
            )
        except TypeError:
            n_recipes = 300

        ingredient = response.xpath("//h1[@class='feed-page__heading ']/text()").get()

        body = {
            "feed": 0,
            "size": n_recipes,
            "slug": ingredient.lower(),
            "type": "ingredient",
        }
        api_quote = url.urlencode(body).replace("+", "-")
        api_req = response.urljoin("/api/proxy/tasty/feed-page?" + api_quote)
        yield scrapy.Request(api_req, self.parse_body)

    def parse_body(self, response):
        queried_recipes = json.loads(response.body)

        for recipe in queried_recipes["items"]:
            recipe_page = url.urljoin("https://tasty.co/recipe/", recipe["slug"])
            yield scrapy.Request(recipe_page, self.parse_recipe)

    def parse_recipe(self, response):
        # yield None
        if response is not None:
            yield parse_json_recipe(response)
