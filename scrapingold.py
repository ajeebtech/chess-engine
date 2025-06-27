############ page 13
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import os
import requests

os.makedirs("data/tournaments", exist_ok=True)

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

base_url = "https://www.chessgames.com"

def dismiss_cookie_banner(driver):
    try:
        consent_button = driver.find_element(By.XPATH, '//div[@class="cc-window"]//a[contains(@class, "cc-btn")]')
        consent_button.click()
        print("Dismissed cookie banner.")
        #time.sleep(1)
    except Exception:
        pass  # Banner not found; continue

# Loop through tournament list pages
for page_num in range(1, 215):
    driver.get(f"{base_url}/perl/tournaments?page={page_num}")
    dismiss_cookie_banner(driver)
    #time.sleep(1)

    tournament_links = driver.find_elements(By.XPATH, '//ol/li/a')
    print(f"Page {page_num}: {len(tournament_links)} tournaments")

    for i in range(len(tournament_links)):
        try:
            tournament_links = driver.find_elements(By.XPATH, '//ol/li/a')
            tournament_link = tournament_links[i]
            href = tournament_link.get_attribute("href")
            print(f"Visiting tournament: {href}")
            dismiss_cookie_banner(driver)
            driver.execute_script("arguments[0].click();", tournament_link)
            #time.sleep(1)
            dismiss_cookie_banner(driver)
            #time.sleep(1)

            # Inside tournament: extract game links
            game_links = driver.find_elements(By.XPATH, '//a[contains(@href, "chessgame?gid=")]')
            for j in range(len(game_links)):
                game_links = driver.find_elements(By.XPATH, '//a[contains(@href, "chessgame?gid=")]')
                game = game_links[j]
                game_href = game.get_attribute("href")
                print(f"  Visiting game: {game_href}")
                dismiss_cookie_banner(driver)
                driver.execute_script("arguments[0].click();", game)
                #time.sleep(1)
                dismiss_cookie_banner(driver)
                #time.sleep(1)

                try:
                    pgn_link = driver.find_element(By.XPATH, '//a[contains(@href, "downloadGamePGN")]')
                    pgn_href = pgn_link.get_attribute("href")
                    if pgn_href.startswith("http"):
                        pgn_url = pgn_href
                    else:
                        pgn_url = base_url + pgn_href
                    game_id = pgn_url.split("gid=")[-1]
                    response = requests.get(pgn_url)
                    with open(f"data/tournaments/pgn_game_{game_id}.pgn", "wb") as f:
                        f.write(response.content)
                    print(f"    Saved PGN for game {game_id}")
                except Exception as e:
                    print(f"    Could not download PGN: {e}")

                driver.back()
                dismiss_cookie_banner(driver)
                #time.sleep(1)

            driver.back()
            dismiss_cookie_banner(driver)
            #time.sleep(1)

        except Exception as e:
            print(f"Error: {e}")
            driver.get(f"{base_url}/perl/tournaments?page={page_num}")
            dismiss_cookie_banner(driver)
            #time.sleep(1)
            continue

driver.quit()
