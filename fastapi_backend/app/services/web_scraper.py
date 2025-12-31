# app/services/web_scraper.py
"""
Web scraper service for nutrition information.
Wraps the selenium-based scraper with proper error handling.
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict, Optional
import time

URL = "https://pphtpc.hpb.gov.sg/web/sgfoodid/tools/food-search"


def _create_driver(headless: bool = True) -> webdriver.Chrome:
    """Create a configured Chrome WebDriver instance."""
    options = Options()
    
    if headless:
        options.add_argument("--headless=new")
    
    # Required for running in server/container environments
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    # Reduce detection
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    
    return driver


def _close_popups(driver: webdriver.Chrome) -> None:
    """Close any popup windows or overlays."""
    # Close extra tabs
    if len(driver.window_handles) > 1:
        main = driver.window_handles[0]
        for h in driver.window_handles:
            if h != main:
                driver.switch_to.window(h)
                driver.close()
        driver.switch_to.window(main)
        time.sleep(0.5)
    
    # Try to close overlay buttons
    try:
        close_buttons = driver.find_elements(
            By.CSS_SELECTOR, 
            "button.close, .btn-close, [aria-label='Close']"
        )
        for btn in close_buttons:
            try:
                driver.execute_script("arguments[0].click();", btn)
            except:
                pass
    except:
        pass


def search_foods(search_term: str, headless: bool = True, max_results: int = 5) -> List[str]:
    """
    Search for food items and return top results.
    
    Args:
        search_term: Food name to search for
        headless: Run browser in headless mode
        max_results: Maximum number of results to return
        
    Returns:
        List of food names from search results
    """
    driver = _create_driver(headless)
    
    try:
        driver.get(URL)
        _close_popups(driver)
        
        # Wait for and fill search bar
        search_box = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "#searchBar"))
        )
        search_box.clear()
        search_box.send_keys(search_term)
        
        _close_popups(driver)
        
        # Click search button
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", search_button)
        
        # Wait for results
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".food-item, .row"))
        )
        
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr"))
        )
        
        # Extract result names
        results = []
        for row in rows[:max_results]:
            try:
                clickable = row.find_element(By.CSS_SELECTOR, "div.text-primary")
                name = clickable.text.strip()
                if name:
                    results.append(name)
            except:
                continue
        
        return results
        
    finally:
        driver.quit()


def get_nutrition_details(
    search_term: str, 
    result_index: int = 0, 
    headless: bool = True
) -> Optional[Dict]:
    """
    Get detailed nutrition information for a specific search result.
    
    Args:
        search_term: Food name to search for
        result_index: Index of result to select (0-based)
        headless: Run browser in headless mode
        
    Returns:
        Dictionary with nutrition data or None if failed
    """
    driver = _create_driver(headless)
    
    try:
        driver.get(URL)
        _close_popups(driver)
        
        # Search
        search_box = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "#searchBar"))
        )
        search_box.clear()
        search_box.send_keys(search_term)
        
        _close_popups(driver)
        
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", search_button)
        
        # Wait for results
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".food-item, .row"))
        )
        
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr"))
        )
        
        # Find clickable elements
        clickable_elements = []
        for row in rows[:5]:
            try:
                clickable = row.find_element(By.CSS_SELECTOR, "div.text-primary")
                clickable_elements.append(clickable)
            except:
                continue
        
        if result_index >= len(clickable_elements):
            return None
        
        # Click selected result
        chosen = clickable_elements[result_index]
        driver.execute_script("arguments[0].scrollIntoView(true);", chosen)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", chosen)
        
        # Extract details
        name_el = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#bordered-container h3"))
        )
        desc_el = driver.find_element(By.CSS_SELECTOR, "#bordered-container p")
        
        name = name_el.text.strip()
        description = desc_el.text.strip()
        
        # Extra info (Food Group, Subgroup, etc.)
        extra_info = {}
        info_rows = driver.find_elements(By.CSS_SELECTOR, "div.row.mb-2")
        for row in info_rows:
            cols = row.find_elements(By.CSS_SELECTOR, "div.col-sm-3, div.col-sm-9")
            if len(cols) >= 2:
                label = cols[0].text.strip().rstrip(":")
                value = cols[1].text.strip()
                if label:
                    extra_info[label] = value
        
        # Nutrition tables
        tables = driver.find_elements(By.XPATH, "//table")
        
        if len(tables) >= 2:
            # Table 1: nutrient names
            nutrient_names = [
                td.text.strip()
                for td in tables[0].find_elements(By.XPATH, ".//tr/td")
            ]
            
            # Table 2: values
            per100g_values = []
            value_rows = tables[1].find_elements(By.XPATH, ".//tr")[1:]  # skip header
            for row in value_rows:
                cols = row.find_elements(By.TAG_NAME, "td")
                if cols:
                    per100g_values.append(cols[0].text.strip())
            
            nutrition = dict(zip(nutrient_names, per100g_values))
        else:
            nutrition = {}
        
        return {
            "name": name,
            "description": description,
            "extra_info": extra_info,
            "nutrition": nutrition
        }
        
    except Exception as e:
        print(f"Scraping error: {e}")
        return None
        
    finally:
        driver.quit()