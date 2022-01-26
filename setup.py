from setuptools import setup, find_packages

with open("requirements.txt", "r") as req_file:
    req = req_file.read()

setup(
    name='Probo2',
    version='1.0',
    #python_requires='>=3.9.5',
    packages=find_packages(),
    url='',
    license='',
    author='jklein94',
    author_email='jklein94@uni-koblenz.de',
    description='Evaluation framework for argumentation solvers',
    install_requires=req,
    entry_points='''
      [console_scripts]
      probo2=src.interface:cli
  ''',
)
