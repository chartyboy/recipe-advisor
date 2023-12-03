import sqlite3
import json
import scrapy
import os
import urllib.parse as url
import re
from tqdm import tqdm
from scrapy.crawler import CrawlerProcess
from scrapy.shell import inspect_response

from capstone_crawlers.spiders.spider_utils import (
    parse_json_recipe,
    replace_newlines,
)


class RecipeSpider(scrapy.Spider):
    name = "recipe"
    url_file = "../notebooks/urls.txt"
    start_url = []
    output = dict()

    custom_settings = {
        "FEEDS": {
            "../notebooks/test_recipes.jl": {"format": "jsonlines", "overwrite": True}
        },
        "LOG_LEVEL": "WARN",
        "AUTOTHROTTLE_DEBUG": False,
        "DOWNLOAD_FAIL_ON_DATALOSS": False,
        "FEED_EXPORT_ENCODING": "utf-8",
        # "CLOSESPIDER_PAGECOUNT": 600,
        # "DOWNLOAD_DELAY": 0.1,
    }

    def start_requests(self):
        self.urls = list()
        with open(self.url_file, "r") as f:
            for line in f:
                self.urls.append(line.strip())
        for url in tqdm(self.urls):
            yield scrapy.Request(url, self.parse_recipe)

    def parse_recipe(self, response):
        # yield None
        if response is not None:
            res = parse_json_recipe(response)
            self.output[response.url] = res
            yield res
