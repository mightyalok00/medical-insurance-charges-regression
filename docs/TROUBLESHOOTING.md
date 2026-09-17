# Troubleshooting

## Streamlit command not found
Install dependencies with `pip install -r requirements.txt`.

## Model file not found
Run the app from the repository root and confirm the `models/` directory is present.

## Altair field error
The dashboard explicitly cleans exported `Unnamed: 0` columns and uses explicit Altair field definitions.

## Notebook data path error
The notebook resolves the project root and loads data from `data/raw/insurance.csv`.
