import os
import time
import zipfile
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# Settings
URL = "https://www.pgnmentor.com/files.html"
DOWNLOAD_DIR = "data/pgnmentor"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Setup headless browser
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load the webpage
driver.get(URL)
time.sleep(2)

# Find all .zip download links
zip_links = driver.find_elements(By.XPATH, '//a[contains(@href, ".zip")]')
print(f"Found {len(zip_links)} zip files.")

# Download and extract each .zip file
for link in zip_links:
    href = link.get_attribute("href")
    filename = href.split("/")[-1]
    zip_path = os.path.join(DOWNLOAD_DIR, filename)

    print(f"Downloading {filename}...")
    try:
        response = requests.get(href)
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        # Extract contents
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)

        # Delete zip file
        os.remove(zip_path)
        print(f"Deleted {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

driver.quit()
print("✅ All files downloaded and extracted.")