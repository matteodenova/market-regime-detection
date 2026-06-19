# Config file for paths (YAML-like format)
# Absolute paths
# %pip install pyyaml python-box

import box
import yaml

yaml_text = """

    workspace_root: "c:/Users/mdenova/OneDrive - KPMG/Desktop/Varie/corso python"

    # Relative paths
    data_dir: "data"
    raw_data: "data/raw/df_historical_prices.csv"
    processed_data: "data/processed/df_features.parquet"
    

    """

cfg_path = yaml.safe_load(yaml_text)
cfg_path = box.Box(cfg_path)


