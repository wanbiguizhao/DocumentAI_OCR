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