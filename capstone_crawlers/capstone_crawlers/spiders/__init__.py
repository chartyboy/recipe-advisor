# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import logging
from capstone_crawlers.spiders import spider_utils

logging.getLogger("scrapy.core.scraper").addFilter(
    lambda x: not x.getMessage().startswith("Scraped from")
)
