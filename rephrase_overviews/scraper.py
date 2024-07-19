import os
import sys
import time
import csv

from selenium.common import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from constants import file_path, fields


def login(driver):
    driver.get('https://quillbot.com/login?returnUrl=/welcome')

    time.sleep(5)

    username_box = driver.find_element(By.NAME, 'username')
    password_box = driver.find_element(By.NAME, 'password')

    username_box.send_keys('sandeep.hvpnl@gmail.com')
    password_box.send_keys('Tz12ep34f1@')

    login_button = driver.find_element(By.CSS_SELECTOR, '[data-testid="login-btn"]')
    login_button.click()

    time.sleep(10)


def rephrase_overviews(driver, rows, output_file_name_suffix):
    driver.get('https://quillbot.com/paraphrasing-tool')

    done_count = 0
    start_time = time.time()
    missed_movies = []

    time.sleep(5)
    filename = f"{file_path}/horror_movies_rephrased_{output_file_name_suffix}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    normal_button = driver.find_element(By.ID, 'Paraphraser-mode-tab-9')
    normal_button.click()

    input_box = driver.find_element(by=By.XPATH, value="//input[@placeholder='e.g., “Like a CEO”']")
    input_box.send_keys("Like a movie overview")

    go_button = driver.find_element(By.CSS_SELECTOR, '[data-testid="pphr/custom_mode/popup/submit_button"]')
    go_button.click()

    time.sleep(2)

    with open(filename, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        output_box = driver.find_element(By.ID, 'paraphraser-output-box')
        for row in rows:
            iteration_start_time = time.time()
            # Locate the text box
            text_box = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, 'paraphraser-input-box'))
            )
            text_box.clear()
            text_box.send_keys(row['overview'])

            paraphrase_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="pphr/input_footer/paraphrase_button"]'))
            )
            actions = ActionChains(driver)
            actions.move_to_element(paraphrase_button).click().perform()

            copied_text = ""
            retries_remaining = 10
            while retries_remaining > 0:
                try:
                    WebDriverWait(driver, 20).until(
                        lambda driver: "Words" in driver.find_element(By.CSS_SELECTOR,
                                                                      '[data-testid="output-bottom-quill-controls-default"]').text
                    )
                    copied_text = output_box.text
                    row['overview'] = copied_text
                    writer.writerow(row)
                    break
                except TimeoutException as t:
                    retries_remaining -= 1
                    if retries_remaining > 0:
                        actions = ActionChains(driver)
                        actions.move_to_element(paraphrase_button).click().perform()
                    else:
                        missed_movies.append(row[''])
                        continue

            # Calcs
            done_count += 1
            iteration_time = time.time() - iteration_start_time
            total_time = time.time() - start_time
            average_time_per_row = total_time / done_count
            estimated_time_left = average_time_per_row * (len(rows) - done_count)
            hours = int(estimated_time_left / 3600)
            mins = int(((estimated_time_left / 3600) % 1) * 60)
            seconds = int(((((estimated_time_left / 3600) % 1) * 60) % 1) * 60)
            # Print calcs
            sys.stdout.write("\r" + " " * 100)  # Clear the line
            sys.stdout.write(f"\rDone {done_count} out of {len(rows)} at {iteration_time:.2f} seconds / row. "
                             f"Total time: {total_time:.2f}s. Estimated time left: {hours}:{mins}:{seconds}.")
            sys.stdout.flush()
            output_box.clear()

            # Wait before the next iteration
            time.sleep(1)

    print(missed_movies)
