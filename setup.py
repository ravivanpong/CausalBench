import setuptools

setuptools.setup(
    include_package_data=True,
    name='causalbench',
    version='0.0.1',
    description='causalbench python module',
    url='https://github.com/ravivanpong/CausalBench',
    author='Ravivanpong',
    author_email='upurw@student.kit.edu',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "tqdm>=4.48.2",
        "numpy>=1.19.1",
        "pandas>=0.22.0",
        "scipy>=1.7.3",
        "scikit-learn>=0.21.1",
        "matplotlib>=2.1.2",
        "networkx>=2.5",
    ],
)