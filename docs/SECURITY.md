# Security Notes

This repository contains no credentials or API keys and does not require secrets for local use.

Good practice:
- never commit passwords, tokens, or `.env` secrets
- keep local virtual environments out of version control
- review third-party dependency updates before deployment
- treat uploaded datasets and model artifacts as project assets, not trusted executable input
