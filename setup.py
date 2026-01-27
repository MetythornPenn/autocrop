import os
from setuptools import setup

def read(*paths):
    with open(os.path.join(*paths), 'r') as f:
        return f.read()
    
requirements = [
    'numpy',
    'opencv-python-headless',
    'onnxruntime'
    # 'matplotlib'
]

setup(
    name='autocrop_kh',
    version='1.1',
    packages=['autocrop_kh'],
    url='https://github.com/MetythornPenn/autocrop_kh.git',
    license='Apache Software License 2.0',
    author = 'Metythorn Penn',
    author_email = 'metythorn@gmail.com',
    keywords='autocrop_kh',
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
