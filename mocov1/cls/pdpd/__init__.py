import os
import sys
PROJECT_DIR= os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.realpath( __file__))
                                    )
                                )
                            )

def append_path():
    sys.path.append(PROJECT_DIR)
    sys.path.append(PROJECT_DIR+"/mocov1")
append_path()
sys.path.append(PROJECT_DIR)
sys.path.append(PROJECT_DIR+"/mocov1")
from mocov1.pp_infer import load_model



from visualdl import LogWriter

# create a log file under `./log/scalar_test/train`
with LogWriter(logdir="./log/scalar_test/train") as writer:
    # use `add_scalar` to record scalar values
    writer.add_scalar(tag="acc", step=1, value=0.5678)
    writer.add_scalar(tag="acc", step=2, value=0.6878)
    writer.add_scalar(tag="acc", step=3, value=0.9878)
# you can also use the following method without using context manager `with`:
"""
writer = LogWriter(logdir="./log/scalar_test/train")

writer.add_scalar(tag="acc", step=1, value=0.5678)
writer.add_scalar(tag="acc", step=2, value=0.6878)
writer.add_scalar(tag="acc", step=3, value=0.9878)

writer.close()
"""