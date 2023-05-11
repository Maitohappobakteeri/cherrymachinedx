import common
from log import (
    log,
    warning,
    error,
    important,
    set_status_state,
    pretty_format,
    ProgressStatus,
)

import argparse
import subprocess
import os
import json
import re
import time
import random
import sys
from bs4 import BeautifulSoup


def wait_for_file(filename):
    max_checks = 20
    while not os.path.isfile(filename):
        max_checks -= 1
        if max_checks <= 0:
            raise RuntimeError("loading file was too slow")
        time.sleep(1)


def courtesy_delay():
    time.sleep(random.randint(10, 200) / 20)


def load_page_thumbnails(page):
    important(f"Starting to load page: {page}")
    url = f"https://api.rule34.xxx/index.php?page=dapi&s=post&q=index&json=1&tags=white_background+-monochrome&pid={page}"
    html_path = os.path.join(common.cache_dir, common.timestamp())
    try:
        subprocess.check_output(
            ["curl", url, "-o", html_path], stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        warning("Failed to load search results")
        return

    wait_for_file(html_path)
    results = json.loads(common.read_file_to_string(html_path))
    set_status_state(ProgressStatus(len(results)))
    for image in results:
        id = image["id"]
        image_url = image["preview_url"]
        meta_path = os.path.join(common.dataset_dir, f"{id}.json")
        image_path = os.path.join(common.dataset_dir, "images", "01", f"{id}.jpg")
        if os.path.isfile(meta_path):
            log(f"Found metadata for {id}, skipping download", repeating_status=True)
            continue
        log(f"Downloading {id}", repeating_status=True)
        courtesy_delay()
        try:
            subprocess.check_output(
                ["wget", image_url, "-O", image_path], stderr=subprocess.DEVNULL
            )
            with open(meta_path, "w") as file:
                json.dump(image, file)
        except subprocess.CalledProcessError:
            warning("Failed to load image")


if __name__ == "__main__":
    # important("Scraping quality posts")
    # topics = load_new_topics()
    # important(f"Found {len(topics)} new topics")
    # for topic in topics:
    # scrape_topic(topic)
    common.ensure_dir(os.path.join(common.dataset_dir, "images"))
    common.ensure_dir(os.path.join(common.dataset_dir, "images", "01"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--pages", type=int, default=1)
    args = parser.parse_args()

    log(pretty_format(args.__dict__))
    for i in range(args.start_page, args.start_page + args.pages):
        courtesy_delay()
        load_page_thumbnails(i)

# Delete invalid with identify dataset/images/01/* 2>&1 | grep "insufficient image data" | sed -r 's/.*(dataset\/images\/.*\.jpg).*/\1/' | xargs rm
