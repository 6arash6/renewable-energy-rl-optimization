# Data Directory Documentation

## Directory Structure

The `data` directory contains all datasets and related files necessary for the Renewable Energy Reinforcement Learning (RL) project. The structure is as follows:

```
data/
├── raw/                # Contains raw data files from various sources
├── processed/          # Contains processed data ready for modeling
└── README.md           # Documentation for the data directory
```

## Data Sources

The data used in this project is sourced from several reputable platforms:

1. **Renewables.ninja**: Provides historical weather data, specifically solar and wind energy data.
2. **ENTSO-E**: Offers electricity consumption and generation data from European countries.
3. **Kaggle Solar**: Contains datasets related to solar energy, including solar panel efficiency and production data.

## Data Processing Pipeline

The data processing pipeline consists of the following steps:

1. **Data Acquisition**: Fetch raw data from the above sources.
2. **Data Cleaning**: Handle missing values and inconsistencies in the datasets.
3. **Data Transformation**: Convert raw data into a usable format for analysis and modeling.
4. **Data Storage**: Store processed data in the `processed` directory for easy access during model training.

## Usage Instructions

To use the data in this project:

1. Clone the repository.
2. Navigate to the `data` directory.
3. Follow the data processing pipeline to prepare the datasets for training your reinforcement learning models.
