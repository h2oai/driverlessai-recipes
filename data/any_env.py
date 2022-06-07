"""Modify dataset with arbitrary env"""

from h2oaicore.data import CustomData


class FreshEnvData(CustomData):

    @staticmethod
    def create_data(X=None):
        return FreshEnvData.create_data_popen(X=X)

    @staticmethod
    def create_data_popen(X=None):
        # Specify the python package dependencies.  Will be installed in order of list
        pyversion = "3.8"
        _install_h2oaicore = False
        _install_datatable = True
        _modules_needed_by_name = ["pandas==1.1.5"]

        import os
        from h2oaicore.data import DataContribLoader
        env_dir = DataContribLoader()._env_dir
        env_path = os.path.abspath(os.path.join(env_dir, "recipe_env"))
        os.makedirs(env_path, exist_ok=True)
        X_file = os.path.abspath(os.path.join(env_path, "X.jay"))
        Y_file = os.path.abspath(os.path.join(env_path, "Y.jay"))

        if X is not None:
            X.to_jay(X_file)

        python_script_file = os.path.abspath(os.path.join(env_path, "script.py"))
        with open(os.path.abspath(__file__), "rt") as src:
            with open(python_script_file, "wt") as dst:
                for line in src.readlines():
                    if line.startswith("from h2oaicore.data import CustomData"):
                        continue
                    if line.endswith("CustomData):\n"):
                        line = line.replace("CustomData", "object")
                    dst.write(line)

        script_name = os.path.abspath(os.path.join(env_path, "script.sh"))
        import os
        with open(script_name, "wt") as f:
            print("set -o pipefail", file=f)
            print("set -ex", file=f)
            print("unset PYTHONPATH", file=f)
            print("unset PYTHONUSERBASE", file=f)
            print("mkdir -p %s" % env_path, file=f)
            print("virtualenv -p python%s %s" % (pyversion, env_path), file=f)
            print("source %s/bin/activate" % env_path, file=f)
            dai_home = os.environ.get('DRIVERLESS_AI_HOME', os.getcwd())
            template_dir = os.environ.get("H2OAI_SCORER_TEMPLATE_DIR", os.path.join(dai_home, 'h2oai_scorer'))
            if _install_h2oaicore:
                print("pip install %s" % os.path.join(template_dir, 'scoring-pipeline', 'license-*'), file=f)
                print("pip install %s" % os.path.join(template_dir, 'scoring-pipeline', 'h2oaicore-*'), file=f)
            if _install_datatable:
                print("pip install %s" % os.path.join(template_dir, 'scoring-pipeline', 'datatable-*'), file=f)
            for pkg in _modules_needed_by_name:
                print("%s/bin/pip install %s --ignore-installed" % (env_path, pkg), file=f)
            print("cd %s" % os.path.dirname(python_script_file), file=f)
            print("%s/bin/python %s --X %s --Y %s" % (env_path, python_script_file, X_file, Y_file), file=f)
        import stat
        os.chmod(script_name, stat.S_IRWXU)

        syntax = [script_name]
        from h2oaicore.systemutils import FunnelPopen
        os.environ.pop('PYTHONPATH', None)
        os.environ.pop('PYTHONUSERBASE', None)
        with FunnelPopen(syntax, shell=True) as fp:
            fp.launch().read()

        import datatable as dt
        Y = dt.fread(Y_file)

        from h2oaicore.systemutils import remove
        remove(X_file)
        remove(Y_file)

        return Y

    @staticmethod
    def create_data_sub(X=None):
        my_path = os.path.dirname(__file__)

        import pandas as pd
        assert pd.__version__ == "1.1.5", "actual: %s" % pd.__version__

        url = "http://data.un.org/_Docs/SYB/CSV/SYB63_226_202009_Net%20Disbursements%20from%20Official%20ODA%20to%20Recipients.csv"
        import urllib.request

        new_file = os.path.join(my_path, "user_file.csv")
        urllib.request.urlretrieve(url, new_file)

        import datatable as dt
        return dt.fread(new_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DAI recipe")
    parser.add_argument(
        "--X",
        dest='X',
        default=None,
        type=str,
        help="input frame path",
    )
    parser.add_argument(
        "--Y",
        dest='Y',
        default=None,
        type=str,
        help="output frame path",
    )
    args = parser.parse_args()

    import os
    import datatable as dt

    if os.path.isfile(args.X):
        X = dt.fread(args.X)
    else:
        X = None
    Y = FreshEnvData.create_data_sub(X)
    Y.to_jay(args.Y)
