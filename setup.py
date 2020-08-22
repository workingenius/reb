from distutils.core import setup
setup(
    name='reb',
    version='0.0.1',
    py_modules=['reb'],
    author='ruqishang',
    author_email='workingenius@163.com',
    url='https://github.com/workingenius/reb',
    description='Regular Expression Beautiful',
    license='MIT',
    extras_require={
        'dev': ['pytest', 'mypy'],
    },
)
