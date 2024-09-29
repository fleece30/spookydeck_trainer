import os
import csv
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service

from cast_scraper.constants import file_path, fields
from cast_scraper.helper import print_progress_bar


def setup():
    gecko_path = '/snap/bin/firefox.geckodriver'
    assert os.path.exists(gecko_path), "GeckoDriver not found at the specified path"
    assert os.access(gecko_path, os.X_OK), "GeckoDriver is not executable"

    profile_path = '/home/fleece/snap/firefox/common/.mozilla/firefox/ulftqqiy.selenium'
    assert os.path.exists(profile_path), "Firefox profile path does not exist"

    profile = webdriver.FirefoxProfile(profile_path)

    options = webdriver.FirefoxOptions()
    options.profile = profile
    options.add_argument("--window-size=1920,1080")

    service = Service(executable_path=gecko_path)
    driver = webdriver.Firefox(service=service, options=options)
    return driver


def get_people(driver, rows, output_file_name_suffix):
    filename = f"{file_path}/horror_movies_rephrased_with_cast_{output_file_name_suffix}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    done_count = 0
    start_time = time.time()

    print_progress_bar(0, len(rows), 'Progress:', 'Complete', length=35)
    with open(filename, 'w+') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            driver.get(f"https://www.themoviedb.org/movie/{row['id']}")

            # Check if error page
            error_box = driver.find_elements(By.CLASS_NAME, 'error_wrapper')
            if len(error_box) != 0:
                continue

            director_box = driver.find_elements(By.CSS_SELECTOR, "ol[class='people no_image']")
            if len(director_box) != 0:
                director_li_elements = director_box[0].find_elements(By.TAG_NAME, "li")
                if len(director_li_elements) == 0:
                    row['director'] = 'NA'
                for creator in range(len(director_li_elements)):
                    if 'Director' in director_li_elements[creator].text:
                        director = director_li_elements[creator].text.split('\n')[0]
                        row['director'] = director

            cast_list_scroller = driver.find_elements(By.CSS_SELECTOR, "ol[class='people scroller']")
            if len(cast_list_scroller) != 0:
                all_li_from_scroller = cast_list_scroller[0].find_elements(By.TAG_NAME, 'li')
                cast_string = ""

                for people_counter in range(len(all_li_from_scroller) - 1):
                    li_links = all_li_from_scroller[people_counter].find_elements(By.TAG_NAME, 'a')
                    cast_string = cast_string + ", " + li_links[-1].text
                row['cast'] = cast_string[2:]

            writer.writerow(row)

            print_progress_bar(i + 1, len(rows), prefix='Progress:', suffix='Complete', length=35)
            time.sleep(2)

    driver.close()

def get_release_dates(driver, rows, output_file_name_suffix):
    for i, row in enumerate(rows):
        driver.get(f"https://www.themoviedb.org/movie/{row['id']}")