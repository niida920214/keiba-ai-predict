import argparse
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scraper import (
    Results, HorseResults, Peds, Return, 
    scrape_kaisai_date, scrape_race_id_list, update_data
)
import modules.constants.local_paths as local_paths

def main():
    parser = argparse.ArgumentParser(description="Update Keiba Data")
    parser.add_argument("--start", type=str, required=True, help="Start Date (YYYY-MM)")
    parser.add_argument("--end", type=str, required=True, help="End Date (YYYY-MM)")
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end
    
    print(f"Update Period: {start_date} to {end_date}")
    
    # 2. Get Race IDs
    print("Fetching Race IDs...")
    try:
        kaisai_dates = scrape_kaisai_date(start_date, end_date)
        race_id_list = scrape_race_id_list(kaisai_dates)
    except Exception as e:
        print(f"Error fetching race list: {e}")
        return

    print(f"Found {len(race_id_list)} races.")
    
    if not race_id_list:
        print("No races found. Exiting.")
        return

    # 3. Scrape Data
    print("Scraping Results...")
    new_results = Results.scrape(race_id_list)
    
    print("Scraping HorseResults...")
    # Get all horse IDs involved
    horse_id_list = new_results["horse_id"].unique().tolist() if not new_results.empty else []
    new_horse_results = HorseResults.scrape(horse_id_list)
    
    print("Scraping Peds...")
    new_peds = Peds.scrape(horse_id_list)
    
    print("Scraping Returns...")
    new_returns = Return.scrape(race_id_list)
    
    # 4. Update Existing Data (Correct Paths)
    print("Updating existing pickle files...")
    
    paths = {
        "results": local_paths.RAW_RESULTS_DIR / "race_results.pickle",
        "horse_results": local_paths.RAW_HORSE_RESULTS_DIR / "horse_results.pickle",
        "peds": local_paths.RAW_PEDS_DIR / "peds.pickle",
        "return_tables": local_paths.RAW_RETURN_TABLES_DIR / "return_tables.pickle",
    }
    
    new_data = {
        "results": new_results,
        "horse_results": new_horse_results,
        "peds": new_peds,
        "return_tables": new_returns,
    }
    
    for key, path in paths.items():
        # Ensure parent dir exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.exists():
            print(f"Updating {key}...")
            try:
                with open(path, "rb") as f:
                    old_df = pickle.load(f)
                
                if isinstance(old_df, pd.DataFrame):
                    updated_df = update_data(old_df, new_data[key])
                    with open(path, "wb") as f:
                        pickle.dump(updated_df, f)
                    print(f"Saved {key} ({len(updated_df)} rows).")
                else:
                    print(f"Warning: {path} does not contain a DataFrame. Overwriting with new data.")
                    with open(path, "wb") as f:
                        pickle.dump(new_data[key], f)
                    print(f"Saved {key} (Overwritten).")
            except Exception as e:
                print(f"Error updating {key}: {e}")
        else:
            print(f"Creating new {key}...")
            with open(path, "wb") as f:
                pickle.dump(new_data[key], f)
            print(f"Saved {key}.")

    print("Data update completed.")

if __name__ == "__main__":
    main()
