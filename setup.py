from setuptools import setup

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='hypers',

    version='0.0.10',

    packages=['hypers',
              'hypers._view',
              'hypers._view._form',
              'hypers._tools',
              'hypers._learning'],

    python_requires='>=3.5.0',

    url='https://github.com/priyankshah7/hypers',

    download_url='https://github.com/priyankshah7/hypers/archive/v0.0.11.tar.gz',

    license='BSD 3-Clause',

    author='Priyank Shah',

    author_email='priyank.shah@kcl.ac.uk',

    description='Hyperspectral data analysis and machine learning',

    long_description=LONG_DESCRIPTION,

    long_description_content_type='text/markdown',

    keywords=['hyperspectral',
              'data-analysis',
              'clustering',
              'matrix-decompositions',
              'hyperspectral-analysis',
              'machine learning'],

    install_requires=['numpy>=1.14.2',
                      'matplotlib',
                      'scipy>=1.1.0',
                      'pyqt5>=5.10.1',
                      'pyqtgraph>=0.10.0',
                      'scikit-learn>=0.20.1']
)
