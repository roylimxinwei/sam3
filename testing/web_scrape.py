from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

URL = "https://pphtpc.hpb.gov.sg/web/sgfoodid/tools/food-search"

def top5_scrape(search_term, headless=False):
    """
    Searches for a food term and returns the top 5 result names.
    
    Args:
        search_term (str): The food name to search for
        headless (bool): Whether to run browser in headless mode
    
    Returns:
        list: List of food names (strings) from the top 5 results
    """
    options = Options()
    if headless:
        options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(URL)

        # ----- Close popup tab if it appears -----
        if len(driver.window_handles) > 1:
            main = driver.window_handles[0]
            for h in driver.window_handles:
                if h != main:
                    driver.switch_to.window(h)
                    driver.close()
            driver.switch_to.window(main)
            time.sleep(1)

        # ----- Wait for the search bar -----
        search_box = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "#searchBar"))
        )

        # ----- Enter search term in the search bar -----
        search_box.clear()
        search_box.send_keys(search_term)

        # ----- Try to close any cookie/popup overlays -----
        try:
            close_buttons = driver.find_elements(By.CSS_SELECTOR, "button.close, .btn-close, [aria-label='Close']")
            for btn in close_buttons:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                except:
                    pass
        except:
            pass

        # ----- Click the Search button -----
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", search_button)

        print("Clicked search. Waiting for results...")

        # ----- Wait for results page to change -----
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".food-item, .row"))
        )

        print("Search results should be visible now.")

        # ----- Wait for table results to load -----
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr"))
        )

        # ----- Extract first 5 result names -----
        results = []

        for row in rows[:5]:
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

def scrape_nutrition_by_index(search_term, chosen_idx=0, output_file="nutrition_results.txt", headless=False):
    """
    Searches for a food term and scrapes nutrition from the result at chosen_idx.
    
    Args:
        search_term (str): The food name to search for
        chosen_idx (int): Index of the result to click (0-based, default 0)
        output_file (str): Optional file path to save results
        headless (bool): Whether to run browser in headless mode
    
    Returns:
        dict: Dictionary containing name, description, extra_info, and nutrition data
    """
    options = Options()
    if headless:
        options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(URL)

        # ----- Close popup tab if it appears -----
        if len(driver.window_handles) > 1:
            main = driver.window_handles[0]
            for h in driver.window_handles:
                if h != main:
                    driver.switch_to.window(h)
                    driver.close()
            driver.switch_to.window(main)
            time.sleep(1)

        # ----- Wait for the search bar -----
        search_box = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "#searchBar"))
        )

        # ----- Enter search term in the search bar -----
        search_box.clear()
        search_box.send_keys(search_term)

        # ----- Try to close any cookie/popup overlays -----
        try:
            close_buttons = driver.find_elements(By.CSS_SELECTOR, "button.close, .btn-close, [aria-label='Close']")
            for btn in close_buttons:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                except:
                    pass
        except:
            pass

        # ----- Click the Search button -----
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", search_button)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", search_button)

        print("Clicked search. Waiting for results...")

        # ----- Wait for results page to change -----
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".food-item, .row"))
        )

        # ----- Wait for table results to load -----
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr"))
        )

        # ----- Find and click the chosen result -----
        clickable_elements = []
        for row in rows[:5]:
            try:
                clickable = row.find_element(By.CSS_SELECTOR, "div.text-primary")
                clickable_elements.append(clickable)
            except:
                continue

        if chosen_idx >= len(clickable_elements):
            print(f"Index {chosen_idx} out of range. Only {len(clickable_elements)} results found.")
            return None

        chosen = clickable_elements[chosen_idx]
        print(f"Clicking on result {chosen_idx + 1}: {chosen.text.strip()}")
        driver.execute_script("arguments[0].scrollIntoView(true);", chosen)
        time.sleep(0.3)
        driver.execute_script("arguments[0].click();", chosen)

        # ----- Extract nutrition information -----
        name_el = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#bordered-container h3")
            )
        )
        desc_el = driver.find_element(
            By.CSS_SELECTOR, "#bordered-container p"
        )

        name = name_el.text.strip()
        description = desc_el.text.strip()

        # --- Key–value rows (Food Group, Food Subgroup, etc.) ---
        info_rows = driver.find_elements(By.CSS_SELECTOR, "div.row.mb-2")

        extra_info = {}
        for row in info_rows:
            cols = row.find_elements(By.CSS_SELECTOR, "div.col-sm-3, div.col-sm-9")
            if len(cols) < 2:
                continue
            label = cols[0].text.strip().rstrip(":")
            value = cols[1].text.strip()
            if label:
                extra_info[label] = value

        # ---- table 1: nutrient names ----
        nutrient_name_table = driver.find_elements(By.XPATH, "//table")[0]
        nutrient_names = [
            row.text.strip()
            for row in nutrient_name_table.find_elements(By.XPATH, ".//tr/td")
        ]

        # ---- table 2: per 100g + per serving values ----
        nutrient_value_table = driver.find_elements(By.XPATH, "//table")[1]

        per100g_values = []
        nutrition_rows = nutrient_value_table.find_elements(By.XPATH, ".//tr")[1:]  # skip header

        for row in nutrition_rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            per100g_values.append(cols[0].text.strip())

        # Map names → per 100g values
        nutrition = dict(zip(nutrient_names, per100g_values))

        result_data = {
            "search_term": search_term,
            "name": name,
            "description": description,
            "extra_info": extra_info,
            "nutrition": nutrition
        }

        # ----- Optionally save to file -----
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Search Term: {search_term}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Name: {name}\n")
                f.write(f"Description: {description}\n\n")
                
                f.write("ADDITIONAL INFO:\n")
                for k, v in extra_info.items():
                    f.write(f"  {k}: {v}\n")
                
                f.write("\nNUTRITION (Per 100 g):\n")
                for k, v in nutrition.items():
                    f.write(f"  {k}: {v}\n")
            print(f"Results saved to {output_file}")

        return result_data

    finally:
        driver.quit()

def scrape_nutrition(search_term, output_file="nutrition_results.txt", headless=False, auto_select=True, choice_callback=None):
    """
    Scrapes nutrition information for a given food search term.
    
    Args:
        search_term (str): The food name to search for
        output_file (str): The file path to save results (default: nutrition_results.txt)
        headless (bool): Whether to run browser in headless mode (default: False)
        auto_select (bool): If True, automatically selects first result. If False, uses choice_callback (default: True)
        choice_callback (callable): A function that takes a list of result names and returns the selected index (0-based)
    
    Returns:
        dict: Dictionary containing name, description, extra_info, and nutrition data
    """
    options = Options()
    if headless:
        options.add_argument("--headless")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(URL)

        # ----- Close popup tab if it appears -----
        if len(driver.window_handles) > 1:
            main = driver.window_handles[0]
            for h in driver.window_handles:
                if h != main:
                    driver.switch_to.window(h)
                    driver.close()
            driver.switch_to.window(main)
            time.sleep(1)

        # ----- Wait for the search bar -----
        search_box = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, "#searchBar"))
        )

        # ----- Enter search term in the search bar -----
        search_box.clear()
        search_box.send_keys(search_term)

        # ----- Click the Search button -----
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.btn-primary"))
        )
        search_button.click()

        print("Clicked search. Waiting for results...")

        # ----- Wait for results page to change -----
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".food-item, .row"))
        )

        print("Search results should be visible now.")

        # ----- Wait for table results to load -----
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tbody tr"))
        )

        # ----- Extract first 5 clickable results -----
        results = []

        for row in rows[:5]:
            try:
                clickable = row.find_element(By.CSS_SELECTOR, "div.text-primary")
                name = clickable.text.strip()
            except:
                continue
            
            results.append({
                "name": name,
                "element": clickable
            })

        if not results:
            print("No results found.")
            return None

        # ----- Handle selection based on mode -----
        if auto_select:
            # Automatically select the first result
            chosen_idx = 0
            print(f"\nAuto-selecting first result: {results[chosen_idx]['name']}")
        else:
            # Use callback function for selection
            if choice_callback:
                result_names = [item['name'] for item in results]
                chosen_idx = choice_callback(result_names)
            else:
                # Default to terminal input
                print("\nSelect one of the search results:")
                for idx, item in enumerate(results, start=1):
                    print(f"{idx}. {item['name']}")
                choice = int(input("\nEnter a number (1–5): ").strip())
                chosen_idx = choice - 1

        chosen = results[chosen_idx]
        print(f"Clicking on: {chosen['name']}")
        chosen["element"].click()

        name_el = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "#bordered-container h3")
            )
        )
        desc_el = driver.find_element(
            By.CSS_SELECTOR, "#bordered-container p"
        )

        name = name_el.text.strip()
        description = desc_el.text.strip()

        # --- Key–value rows (Food Group, Food Subgroup, etc.) ---
        info_rows = driver.find_elements(By.CSS_SELECTOR, "div.row.mb-2")

        extra_info = {}
        for row in info_rows:
            cols = row.find_elements(By.CSS_SELECTOR, "div.col-sm-3, div.col-sm-9")
            if len(cols) < 2:
                continue
            label = cols[0].text.strip().rstrip(":")
            value = cols[1].text.strip()
            if label:
                extra_info[label] = value

        print("\n=== BASIC INFO ===")
        print("Name:", name)
        print("Description:", description)
        for k, v in extra_info.items():
            print(f"{k}: {v}")

        # Wait for nutrition table body
        # ---- table 1: nutrient names ----
        nutrient_name_table = driver.find_elements(By.XPATH, "//table")[0]
        nutrient_names = [
            row.text.strip()
            for row in nutrient_name_table.find_elements(By.XPATH, ".//tr/td")
        ]

        # ---- table 2: per 100g + per serving values ----
        nutrient_value_table = driver.find_elements(By.XPATH, "//table")[1]

        per100g_values = []
        rows = nutrient_value_table.find_elements(By.XPATH, ".//tr")[1:]   # skip header

        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            per100g_values.append(cols[0].text.strip())  # 1st column = per 100g

        # Map names → per 100g values
        nutrition = dict(zip(nutrient_names, per100g_values))
        print("\n=== NUTRITION (Per 100 g) ===")
        for k, v in nutrition.items():
            print(f"{k}: {v} per 100 g")

        # ----- Save to file -----
        result_data = {
            "search_term": search_term,
            "name": name,
            "description": description,
            "extra_info": extra_info,
            "nutrition": nutrition
        }

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Search Term: {search_term}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Name: {name}\n")
            f.write(f"Description: {description}\n\n")
            
            f.write("ADDITIONAL INFO:\n")
            for k, v in extra_info.items():
                f.write(f"  {k}: {v}\n")
            
            f.write("\nNUTRITION (Per 100 g):\n")
            for k, v in nutrition.items():
                f.write(f"  {k}: {v}\n")

        print(f"\nResults saved to {output_file}")
        
        return result_data

    finally:
        driver.quit()


if __name__ == "__main__":
    # Example usage
    search_term = input("Enter food name to search: ").strip()
    if search_term:
        scrape_nutrition(search_term, auto_select=True)
