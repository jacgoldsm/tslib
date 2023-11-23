# Building documentation
- Clone repo
- Install mkdocs et al. into venv:
    $ python3 -m venv venv
    $ source venv/bin/activate
    (venv) $ python -m pip install mkdocs
    (venv) $ python -m pip install "mkdocstrings[python]"
    (venv) $ python -m pip install mkdocs-material
- Make changes
- Test documentation page with `mkdocs serve`
- If changes are all good and you are putting in production:
    - Push your changes to main, etc
    - Make a *separate* directory for the site assets, e.g. in the parent directory:
        - `mkdocs build -d ../tslib-site`
    - Deploy to GitHub based on that directory:
        - `mkdocs gh-deploy -d ../tslib-site`