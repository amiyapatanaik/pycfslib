from setuptools import setup

setup(name='pycfslib',
      version='1.1',
      description='Library to read, write amd create CFS file and stream, now supports the NEO sleep staging system.',
      url='https://github.com/neurobittechnologies/pycfslib',
      author='Amiya Patanaik',
      author_email='amiya@neurobit.io',
      license='GPL',
      packages=['pycfslib'],
      install_requires=[
          'numpy',
          'scikit-image',
          'scipy',
          'numba',
      ],
      zip_safe=False)
