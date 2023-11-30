import sqlite3
import json
import scrapy
import os
import urllib.parse as url
import re
from scrapy.crawler import CrawlerProcess
from scrapy.shell import inspect_response

from capstone_crawlers.spiders.spider_utils import parse_json_recipe


class FoodNetworkSpider(scrapy.Spider):
    name = "foodnetwork"

    @classmethod
    def update_settings(cls, settings):
        super().update_settings(settings)
        # enabled_ext = {"scrapy.extensions.closespider.CloseSpider": 100}
        # settings.set("EXTENSIONS", {})
        settings.set("CLOSESPIDER_ITEMCOUNT", 10, priority="spider")
        settings.set("FEEDS", {"test.jl": {"format": "jsonlines", "overwrite": True}})

    def start_requests(self):
        url = "https://www.foodnetwork.com/recipes/recipes-a-z/123"
        yield scrapy.Request(url, self.parse_a_z)

    def parse_a_z(self, response):
        # Capture all category items from A-Z
        for heading in response.xpath(
            "//li[contains(@class,'o-IndexPagination__a-ListItem')]/a/@href"
        ):
            next_page = response.urljoin(heading.get())
            if next_page is not None:
                yield scrapy.Request(next_page, self.parse_categories)

    def parse_categories(self, response):
        # Check status of next button
        # If it is disabled, there are no more pages left to explore for this
        # heading
        # Xpath should only match one item, the next page button
        next_button = response.xpath(
            "//a[contains(@class,'o-Pagination__a-NextButton')]"
        )[0]
        button_class = re.sub(r"(\n\s+){1,}", ".", next_button.attrib["class"].strip())
        next_is_enabled = button_class.split(".")[-1] != "is-Disabled"

        # Loop through pagination
        # Get all the recipes on this page for the letter heading (A-Z)
        for category in response.xpath(
            "//li[contains(@class,'m-PromoList__a-ListItem')]/a/@href"
        ):
            next_page = response.urljoin(category.get())
            yield scrapy.Request(next_page, self.parse_recipe)

        # Move to next page
        if next_is_enabled:
            yield response.follow(next_button.attrib["href"], self.parse_categories)
        else:
            pass

    def parse_recipe(self, response):
        if response is not None:
            yield parse_json_recipe(response)
