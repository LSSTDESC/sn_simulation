from setuptools import setup

# get the version here
pkg_vars = {}

with open("version.py") as fp:
    exec(fp.read(), pkg_vars)

setup(
    name='sn_simulation',
    version=pkg_vars['__version__'],
    description='Simulations for supernovae',
    url='http://github.com/lsstdesc/sn_simulation',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    packages=['sn_simulator', 'sn_simu_wrapper', 'sn_simu_input'],
    # All files from folder sn_simu_input
    package_data={'sn_simu_input': ['*.txt']},
    python_requires='>=3.5',
    zip_safe=False,
    install_requires=[
        'sn_tools>=0.1',
        'dustmaps'
    ],
)
