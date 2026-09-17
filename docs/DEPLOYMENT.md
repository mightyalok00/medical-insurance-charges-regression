# Deployment Guide

## Local Streamlit
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
1. Push the repository to GitHub.
2. Create a Streamlit app from the repository.
3. Set `app.py` as the main file.
4. Use the repository `requirements.txt` for dependencies.

The app uses paths relative to `app.py`, so no machine-specific path is required.
