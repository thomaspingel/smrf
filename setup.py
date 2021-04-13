from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'The Simple Morphological Filter (SMRF)'
LONG_DESCRIPTION = 'SMRF is a binary lidar ground/object classifier. https://doi.org/10.1016/j.isprsjprs.2012.12.002'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="smrf", 
        version=VERSION,
        author="Thomas Pingel",
        author_email="thomas.pingel@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
