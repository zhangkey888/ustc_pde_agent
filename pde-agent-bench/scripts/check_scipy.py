try:
    import scipy
    print("scipy is available")
    import scipy.spatial
    print("scipy.spatial is available")
except ImportError:
    print("scipy is NOT available")
