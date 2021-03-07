from setuptools import setup

setup(
    name='optyshmopty',
    version='0.0.1',
    description='Little library with different optimization methods.',
    author='Maxim Manainen',
    author_email="maximmanainen@gmail.com",
    packages=['optyshmopty', 'optyshmopty.continuous'],
    install_requires=['numpy>=1.12'],
    keywords=[ 'Convex optimization', 'numerical optimization',
              'Python'],
    url='https://github.com/amkatrutsa/liboptpy',
    license='MIT',
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.5'],
)
