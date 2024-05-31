from setuptools import find_packages, setup

setup(
    name="assisting_bounded_humans",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "gymnasium",
        "imitation",
    ],
    extras_require={
        "dev": ["jupyter", "pytest", "black", "isort"],
    },
    description=".",
    author="Davis Foote",
    url="",
    author_email="davisjfoote@gmail.com",
    keywords="",
    license="MIT",
    long_description="",
    long_description_content_type="text/markdown",
    version="0.0.1",
)
