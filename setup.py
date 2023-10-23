from setuptools import setup, find_packages

setup(
    name='Maturarbeit',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0.0',
        'matplotlib>=3.0.0',
        'seaborn>=0.10.0',
        'scikit-learn>=0.20.0',
        'xgboost>=1.0.0',
        'joblib>=0.14.0',
        'ibapi>=9.76.1',
        'yfinance>=0.1.55'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
)

