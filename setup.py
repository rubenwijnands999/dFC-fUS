from setuptools import setup, find_packages

setup(
    name='dFC-fUS',
    version='0.0.1',
    url='https://github.com/rubenwijnands999/dFC-fUS.git',
    license='MIT',
    author='Ruben Wijnands',
    author_email='rubenwijnands999@gmail.com',
    description='Dynamic functional connectivity analysis for functional ultrasound data',
    python_requires='>=3.7, <4',
    
    install_requires=[
        'autograd == 1.4',
        'colorcet == 3.0.0',
        'dash == 2.6.1',
        'dash_bootstrap_components == 1.2.1',
        'dash_core_components == 2.0.0',
        'dash_html_components == 2.0.0',
        'dash_player == 0.0.1',
        'Flask == 2.2.2',
        'h5py == 3.7.0',
        'matplotlib == 3.5.3',
        'numba',
        'numpy == 1.20.3',
        'opencv_python == 4.6.0.66',
        'pandas == 1.0.5',
        'plotly == 5.10.0',
        'pybasicbayes == 0.2.2',
        'tables==3.6.1',
        'scikit_learn == 1.0',
        'scipy == 1.7.1',
        'seaborn == 0.12.0',
        'setuptools',
        'statsmodels == 0.13.2',
        'tqdm == 4.64.1'
    ],    
)
