from setuptools import setup

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='scikit-hyper',

    version='0.0.6',

    packages=['skhyper',
              'skhyper.view',
              'skhyper.view._form',
              'skhyper.cluster',
              'skhyper.process',
              'skhyper.decomposition'],

    python_requires='>=3.5.0',

    url='https://github.com/priyankshah7/scikit-hyper',

    download_url='https://github.com/priyankshah7/scikit-hyper/archive/v0.0.6.tar.gz',

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
                      'scipy>=1.1.0',
                      'matplotlib>=2.2.2',
                      'pyqt5>=5.10.1',
                      'pyqtgraph>=0.10.0',
                      'scikit-learn>=0.19.1',
                      'seaborn>=0.8.0']
)
