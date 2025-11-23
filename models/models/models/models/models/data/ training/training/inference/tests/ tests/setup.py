from setuptools import setup, find_packages

setup(
    name="neurosparse-transformers",
    version="1.0.0",
    description="NeuroSparse Transformers: Spiking-Inspired Sparse Attention for Real-Time Multimodal Streams",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
)
