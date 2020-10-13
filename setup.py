from pathlib import Path
from setuptools import setup, find_packages
from Cython.Build import cythonize

projdir = Path(__file__).parent
readme = (projdir / 'README.md').read_text()

setup(
    name='reb',
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    author='ruqishang',
    author_email='workingenius@163.com',
    url='https://github.com/workingenius/reb',
    description='Regular Expression Beautiful',
    license='MIT',
    extras_require={
        'dev': ['pytest', 'mypy'],
    },
    entry_points={
        'console_scripts': ['reb=reb.cli:main']
    },
    long_description=readme,
    long_description_content_type='text/markdown',
    ext_modules=cythonize("reb/vm2.pyx")
)
