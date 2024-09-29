import time

from selenium.webdriver.common.by import By


def fetch_ids_from_imdb(driver):
    movie_ids = []

    driver.get(
        "https://www.imdb.com/search/title/?title_type=feature&release_date=2023-01-01,"
        "2024-07-21&genres=horror&sort=release_date,desc")

    count = 0
    while True:
        see_more_button = driver.find_elements(By.CLASS_NAME, "ipc-see-more__text")
        if not see_more_button:
            break
        for button in see_more_button:
            driver.execute_script("arguments[0].click();", button)
            count += 1
            print(f"Pressed button {count}/46 times", end="\r")
            time.sleep(3)

    movies = driver.find_elements(By.CSS_SELECTOR,
                                  "ul[class='ipc-metadata-list ipc-metadata-list--dividers-between sc-748571c8-0 "
                                  "jmWPOZ detailed-list-view ipc-metadata-list--base']")

    check_input = input("Do you want to continue?")

    if check_input == "y":
        if len(movies) != 0:
            all_li_from_scroller = movies[0].find_elements(By.TAG_NAME, 'li')
            print(len(all_li_from_scroller))

            for movie_counter in range(len(all_li_from_scroller)):
                li_links = all_li_from_scroller[movie_counter].find_elements(By.CLASS_NAME, 'ipc-title-link-wrapper')
                href = li_links[0].get_attribute('href')
                current_id = href.split('/')[4]
                movie_ids.append(current_id)

        with open('movie_ids.txt', 'w') as file:
            for movie_id in movie_ids:
                file.write(movie_id + '\n')
        driver.close()


def remove_duplicates():
    with open('movie_ids.txt', 'r') as file:
        movie_ids = file.readlines()
        movie_ids_set = set(movie_ids)

    with open('movie_ids_unique.txt', 'w') as file:
        for movie_id in movie_ids_set:
            file.write(movie_id)
