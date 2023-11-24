from setuptools import setup, find_packages


reqs = ["pandas"]
extras = {"spark":"pyspark"}
setup(
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=reqs,
    extras_require = extras,
    name="tslib",
    version="0.0.1",
    description="Complete and consistent API for Time-Series functionality",
    author="Jacob Goldsmith",
    author_email="jacobg314@hotmail.com",
    url="https://github.com/jacgoldsm/tslib",
)
