import os
import pathlib

# define relative paths
DIR = str(pathlib.Path(__file__).parent.parent.absolute())
print(DIR)
print(type(DIR))
SRC_DIR = os.path.join(DIR, "docs/source")
print(SRC_DIR)
FEAT_TUT_DIR = os.path.join(DIR, "tutorials", "feature-use")
APP_TUT_DIR = os.path.join(DIR, "tutorials", "application-example")