from setuptools import setup

setup(
    name='scikit-hyper',

    version='0.0.1a',

    packages=['skhyper',
              'skhyper.view',
              'skhyper.view._form',
              'skhyper.cluster',
              'skhyper.process',
              'skhyper.decomposition'],

    python_requires='>=3.5.0',

    url='https://github.com/priyankshah7/scikit-hyper',

    download_url='https://github.com/priyankshah7/scikit-hyper/archive/v0.0.1.tar.gz',

    license='BSD 3-Clause',

    author='Priyank Shah',

    author_email='priyank.shah@kcl.ac.uk',

    description='Hyperspectral data analysis and machine learning',

    keywords=['hyperspectral',
              'data-analysis',
              'clustering',
              'matrix-decompositions',
              'hyperspectral-analysis',
              'machine learning'],

    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'pyqt5',
                      'pyqtgraph',
                      'scikit-learn']
)
