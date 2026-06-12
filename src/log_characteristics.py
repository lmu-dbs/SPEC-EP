from contextppm.dataset.ECDataset import ECDataset
from contextppm.util.config_utils import read_config
import os
import yaml

from util.paths import CONFIG_PATH, EXPORT_PATH


def main():

    general_config = read_config(os.path.join(CONFIG_PATH, "general_config.yml"))
    data_configs = read_config(os.path.join(CONFIG_PATH, "data_configs.yml"))

    for dataset in general_config["dataset"]:
        try:
            data_config = data_configs[dataset]
        except KeyError as e:
            e.args = (f"desired datset {dataset} not found in data_config.yml",)
            raise

        data = ECDataset.from_csv(
            load_path=data_config["load_path"],
            case_identifier=data_config["case_identifier"],
            activity_identifier=data_config["activity_identifier"],
            timestamp_identifier=data_config["timestamp_identifier"],
        )

        log_characteristics = data.get_characteristics()

        with open(
            os.path.join(
                EXPORT_PATH, "characteristics", f"{dataset}_characteristics.csv"
            ),
            "w",
        ) as f:
            yaml.dump(log_characteristics, f, sort_keys=False)


if __name__ == "__main__":
    main()
