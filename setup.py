from setuptools import setup, find_packages
setup(
    name="us_visa",
    version="0.0.0",
    author_email="panchalganubhav@gmail.com",
    packages=find_packages()  # to find all the packages in the project directory
)

# here find_packages will find the __init__.py file
# in the directory and will consider it as a package