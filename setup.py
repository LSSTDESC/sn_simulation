from setuptools import setup


setup(
    name='sn_simulation',
    version='0.1',
    description='Simulations for supernovae',
    url='http://github.com/lsstdesc/sn_simulation',
    author='Philippe Gris',
    author_email='philippe.gris@clermont.in2p3.fr',
    license='BSD',
    packages=['sn_simulator', 'sn_simu_wrapper'],
    python_requires='>=3.5',
    zip_safe=False,
    install_requires=[
        'sn_tools>=0.1',
        'sn_stackers>=0.1',
        'dustmaps'
    ],
)
