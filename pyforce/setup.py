from setuptools import setup, find_packages
from os.path import abspath, dirname, join

# Fetches the content from README.md
# This will be used for the "long_description" field.
README_MD = open(join(dirname(abspath(__file__)), "../README.md")).read()

setup(
    name='pyforce',
    version='0.1.3',    
    description='Python Framework data-driven model Order Reduction for multi-physiCs problEms',
    long_description=README_MD,
    long_description_content_type="text/markdown",
    url='https://github.com/ROSE-Polimi/pyforce',
    author='Stefano Riva, Carolina Introini, Antonio Cammi',
    author_email='stefano.riva@polimi.it, carolina.introini@polimi.it, antonio.cammi@polimi.it',
    # requires=['python==3.10', 'numpy<=1.23.5',
    # 'numpy', 'scipy', 'tqdm', 'matplotlib',
    # 'pyvista',
    # 'dolfinx', 'mpi4py', 'petsc4py', 'h5py', 'ufl',
    # 'fluidfoam'],
    # install_requires=['numpy', 'scipy', 'tqdm', 'matplotlib', 'pyvista',
    #     'dolfinx', 'mpi4py', 'petsc4py', 'h5py', 'ufl',
    #     'fluidfoam'],
    # tests_require=['numpy', 'scipy', 'tqdm', 'matplotlib', 'pyvista',
    #     'dolfinx', 'mpi4py', 'petsc4py', 'h5py', 'ufl',
    #     'fluidfoam'],    
    license='MIT',
    packages=find_packages(exclude="./tests"),
    keywords="reduced-order-modelling, data-assimilation, dolfinx",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.10',
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)    
