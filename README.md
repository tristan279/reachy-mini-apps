# Reachy Mini Apps

A monorepo containing multiple Reachy Mini applications.

## Structure

Each app is in its own subdirectory:
- `test_hello_world/` - Test Hello World app

## Installing an App

To install a specific app on the Reachy Mini bot, SSH into the bot and run:

```bash
pip install git+https://github.com/YOUR_USERNAME/reachy-mini-apps.git#subdirectory=test_hello_world
```

Replace:
- `YOUR_USERNAME` with your GitHub username
- `test_hello_world` with the app directory name you want to install

## Adding a New App

1. Create a new directory for your app
2. Set up the app structure with `pyproject.toml` and entry points
3. Follow the same structure as `test_hello_world/`









