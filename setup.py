from setuptools import setup, find_packages
setup(
    name='reb',
    version='0.0.4',
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
)
