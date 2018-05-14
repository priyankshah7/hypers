=====
Tools
=====

`scikit-hyper` currently contains the following tools:

- :class:`~skhyper.process.normalize`
    This function allows you to perform spectral normalization on the entire
    dataset. This is useful if your spectral data contains a known multiplicative
    noise source of spectral feature.

- :class:`~skhyper.process.savgol_filter`
    This function implements ``scipy``'s version of the Savitzky-Golay filter, a
    filter commonly used to smooth spectra. The usual parameters are available here.