import sqlite3
import json
import scrapy
import os
import urllib.parse as url
import re
from scrapy.crawler import CrawlerProcess
from scrapy.shell import inspect_response

from capstone_crawlers.spiders.spider_utils import (
    parse_json_recipe,
    replace_newlines,
)


class EpicuriousSpider(scrapy.Spider):
    name = "epicurious"
    custom_settings = {
        "FEEDS": {"epicurious.jl": {"format": "jsonlines", "overwrite": True}},
        "LOG_LEVEL": "INFO",
        "AUTOTHROTTLE_DEBUG": False,
        # "CLOSESPIDER_PAGECOUNT": 600,
        # "DOWNLOAD_DELAY": 0.1,
    }

    def start_requests(self):
        url = "https://www.epicurious.com/search?page=1&sort=highestRated"
        yield scrapy.Request(url, self.parse_page)

    def parse_page(self, response):
        # Capture all tile items
        for tile in response.xpath("//h4[@class='hed']/a/@href"):
            recipe = response.urljoin(tile.get())
            if recipe is not None:
                yield scrapy.Request(recipe, self.parse_recipe)

        # Move to next pages
        n_pages = int(response.xpath("//a[@class='last-page']/text()").get())
        for i in range(2, n_pages + 1):
            body = {"page": str(i), "sort": "highestRated"}
            search_quote = url.urlencode(body)
            search_req = response.urljoin("/search?" + search_quote)
            yield scrapy.Request(search_req, self.parse_next_pages)

    def parse_next_pages(self, response):
        # print(response.url)
        for tile in response.xpath("//h4[@class='hed']/a/@href"):
            recipe = response.urljoin(tile.get())
            # print(recipe)
            if recipe is not None:
                yield scrapy.Request(recipe, self.parse_recipe)

    def parse_recipe(self, response):
        # yield None
        if response is not None:
            yield parse_json_recipe(response)
