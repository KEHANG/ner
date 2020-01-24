from setuptools import find_packages, setup

exec(open('tagger/version.py').read())

setup(
    name='ner',
    version=ner_version,
    packages=find_packages(exclude=["tests"]),
    description='A Deep Learning Based Name Entity Recognition Tagger',
    author='Kehang Han',
    author_email='kehanghan@gmail.com'
)