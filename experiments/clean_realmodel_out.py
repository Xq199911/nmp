"""
Clean previous realmodel output dirs for injection ON/OFF to ensure paired attn files.
"""
import os
import shutil

def main():
    base = os.path.join("experiments", "realmodel_out")
    for name in ("inject_on_attn", "inject_off_attn"):
        path = os.path.join(base, name)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print("removed", path)
            except Exception as e:
                print("failed to remove", path, e)
        else:
            print("not exist", path)

if __name__ == "__main__":
    main()


