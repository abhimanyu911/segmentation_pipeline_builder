from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]

setup(
  name='dataset_builder_for_segmentation',
  version='0.0.3',
  description='A dataset pipeline builder for semantic image segmentation',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='https://github.com/abhimanyu911/segmentation_pipeline_builder',  
  author='Abhimanyu Borthakur',
  author_email='abhimanyuborthakur@hotmail.com',
  license='MIT', 
  classifiers=classifiers,
  keywords='dataset builder', 
  packages=find_packages(),
  install_requires=[] 
)