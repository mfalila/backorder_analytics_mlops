from setuptools import find_packages, setup


setup(
    name='src',
    packages=find_packages(include=[
        'build_library',
        'prediction_service',
        'src.data']),
    version='0.1.0',
    description='End to End Machine learning pipeline with MLOps tools',
    author='Sam Mfalila',
    author_email="sam.mfalila@gmail.com",
    license='MIT',
)
