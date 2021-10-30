from setuptools import setup, find_packages
import os
import codecs


here = os.path.abspath(os.path.dirname(__file__))

# load README and render as markdown
#with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#    long_description = "\n" + fh.read()


# simpler and hopefully works just as well
with open("README.md") as fh:
    long_description = "\n" + fh.read()


VERSION = '0.0.1'
DESCRIPTION = 'example upload'
LONG_DESCRIPTION = 'A package that allows you to...'

# Setting up
setup(
    name="basicFileTest",
    version=VERSION,
    author="Adam",
    author_email="<adam@adam.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    py_modules=['main'],      # file names
    package_dir={'':'src'},
    #packages=find_packages(),
    install_requires=['numpy'],
    keywords=['python', 'example'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)