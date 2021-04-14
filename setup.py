from setuptools import setup, find_packages

setup(
    name='smrf',
    version='1.1.0',
    packages=['smrf',],
    license='MIT',
    long_description='SMRF is a binary lidar ground/object classifier.',
    url='https://doi.org/10.1016/j.isprsjprs.2012.12.002',

    author='Thomas Pingel',
    author_email='thomas.pingel@gmail.com',

    classifiers=['Development Status :: 4 - Beta',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Geographic Information Science',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'License :: OSI Approved :: MIT License'],
    keywords='GIS lidar',
	install_requires=['scipy','pandas','rasterio','numpy'],

	)