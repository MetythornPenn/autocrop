import os
from setuptools import setup

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()
    
requirements = [
    'torch',
    'torchvision',
    'numpy',
    'opencv-python-headless',
    # 'matplotlib'
]

setup(
    name='autocrop',
    version='0.0.1',
    packages=['autocrop'],
    url='https://github.com/MetythornPenn/autocrop.git',
    license='Apache Software License 2.0',
    author = 'Metythorn Penn',
    author_email = 'metythorn@gmail.com',
    keywords='autocrop',
    description='Document Extraction Inference API using DeepLabV3 with Pretrain Model',
    install_requires=requirements,
    long_description=(read('README.md')),
    long_description_content_type='text/markdown',
	classifiers= [
		'Natural Language :: English',
		'License :: OSI Approved :: Apache Software License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
	],
)