from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='TrackEval',
    url='https://github.com/JonathonLuiten/TrackEval',
    # Needed to actually package something
    packages=find_packages(include=['trackeval', 'trackeval.*']),
    # Needed for dependencies
    install_requires=['numpy==1.20.2','scipy==1.4.1'],
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='Package with trackeval kit to evaluate object tracking tasks',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
