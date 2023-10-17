import subprocess
import os

parent_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = parent_directory.replace("\\", "/")
is_train = "True"
subprocess.run(['Python', parent_directory+'/feature_engineering.py', is_train])
subprocess.run(['Python', parent_directory+'/train.py'])
