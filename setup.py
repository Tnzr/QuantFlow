from setuptools import setup, find_packages

setup(
    name="quantflow",
    version="0.1.0",
    description="QuantFlow: Options screening, rule-based recommendations, and portfolio analytics platform.",
    author="Bryan Garcia Rodriguez",
    author_email="bryan.garcia.b9r@gmail.com",
    packages=find_packages(),
    install_requires=[],  # requirements handled by environment.yml/requirements.txt
    include_package_data=True,
    python_requires=">=3.11",
)
