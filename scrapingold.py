############ page 13
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import time
import os
import requests
import re

os.makedirs("data/tournaments", exist_ok=True)

options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

base_url = "https://www.chessgames.com"

def dismiss_cookie_banner(driver):
    try:
        consent_button = driver.find_element(By.XPATH, '//div[@class="cc-window"]//a[contains(@class, "cc-btn")]')
        consent_button.click()
        print("Dismissed cookie banner.")
        time.sleep(0.5)
    except Exception:
        pass  # Banner not found; continue

def find_pgn_download_link(driver):
    """Try multiple methods to find the PGN download link"""
    selectors = [
        '//a[contains(@href, "downloadGamePGN")]',
        '//a[contains(text(), "Download PGN")]',
        '//a[contains(text(), "PGN")]',
        '//a[contains(@href, "pgn")]',
        '//a[contains(@href, "download")]'
    ]
    
    for selector in selectors:
        try:
            element = driver.find_element(By.XPATH, selector)
            href = element.get_attribute("href")
            if href and ("pgn" in href.lower() or "download" in href.lower()):
                return href
        except NoSuchElementException:
            continue
    
    # If no link found, try to find any link that might contain the game ID
    try:
        current_url = driver.current_url
        game_id_match = re.search(r'gid=(\d+)', current_url)
        if game_id_match:
            game_id = game_id_match.group(1)
            # Try to construct the PGN URL directly
            pgn_url = f"{base_url}/perl/downloadGamePGN?gid={game_id}"
            return pgn_url
    except Exception:
        pass
    
    return None

def save_game_pgn(driver, game_href):
    """Attempt to save PGN for a specific game"""
    try:
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find PGN download link
        pgn_url = find_pgn_download_link(driver)
        
        if not pgn_url:
            print(f"    No PGN download link found for {game_href}")
            # Save page source for debugging
            game_id_match = re.search(r'gid=(\d+)', game_href)
            if game_id_match:
                game_id = game_id_match.group(1)
                with open(f"data/tournaments/debug_page_{game_id}.html", "w", encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"    Saved debug page for game {game_id}")
            return False
        
        # Extract game ID from URL
        game_id_match = re.search(r'gid=(\d+)', pgn_url)
        if not game_id_match:
            game_id_match = re.search(r'gid=(\d+)', game_href)
        
        if game_id_match:
            game_id = game_id_match.group(1)
        else:
            game_id = "unknown"
        
        # Download PGN
        if pgn_url.startswith("http"):
            full_url = pgn_url
        else:
            full_url = base_url + pgn_url
        
        response = requests.get(full_url, timeout=30)
        response.raise_for_status()
        
        # Check if response contains PGN data
        content = response.text
        if not content.strip() or len(content) < 50:
            print(f"    Empty or invalid PGN content for game {game_id}")
            return False
        
        with open(f"data/tournaments/pgn_game_{game_id}.pgn", "w", encoding='utf-8') as f:
            f.write(content)
        print(f"    Saved PGN for game {game_id}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"    Network error downloading PGN: {e}")
        return False
    except Exception as e:
        print(f"    Error saving PGN: {e}")
        return False

# Loop through tournament list pages
for page_num in range(13, 215):
    try:
    driver.get(f"{base_url}/perl/tournaments?page={page_num}")
    dismiss_cookie_banner(driver)
        time.sleep(1)

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
                time.sleep(1)
            dismiss_cookie_banner(driver)
                time.sleep(1)

            # Inside tournament: extract game links
            game_links = driver.find_elements(By.XPATH, '//a[contains(@href, "chessgame?gid=")]')
                print(f"  Found {len(game_links)} games in tournament")
                
            for j in range(len(game_links)):
                    try:
                game_links = driver.find_elements(By.XPATH, '//a[contains(@href, "chessgame?gid=")]')
                game = game_links[j]
                game_href = game.get_attribute("href")
                print(f"  Visiting game: {game_href}")
                dismiss_cookie_banner(driver)
                driver.execute_script("arguments[0].click();", game)
                        time.sleep(1)
                        dismiss_cookie_banner(driver)
                        time.sleep(1)

                        success = save_game_pgn(driver, game_href)
                        if not success:
                            print(f"    Failed to save PGN for {game_href}")

                        driver.back()
                dismiss_cookie_banner(driver)
                        time.sleep(1)

                    except Exception as e:
                        print(f"    Error processing game: {e}")
                        # Try to go back to tournament page
                        try:
                            driver.back()
                            dismiss_cookie_banner(driver)
                            time.sleep(1)
                        except:
                            # If back fails, reload tournament page
                            driver.get(href)
                            dismiss_cookie_banner(driver)
                            time.sleep(1)
                        continue

                driver.back()
                dismiss_cookie_banner(driver)
                time.sleep(1)

        except Exception as e:
                print(f"Error processing tournament: {e}")
            driver.get(f"{base_url}/perl/tournaments?page={page_num}")
            dismiss_cookie_banner(driver)
                time.sleep(1)
                continue
                
    except Exception as e:
        print(f"Error on page {page_num}: {e}")
            continue

driver.quit()
