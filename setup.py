import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name="FID_Simulation", # Replace with your own username
    version="0.0.1",
    author="Rene Reimann    ",
    author_email="rreimann@uni-mainz.de",
    description="A package to simulate Free Induction Decay Signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/renereimann/FID_Simulation",
    license=license,
    packages=setuptools.find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
