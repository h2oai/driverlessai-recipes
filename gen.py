exclude = ['.', '.git', 'data', 'Makefile', 'LICENSE', 'README.md', 'gen.sh', 'gen.py']
import os
print("# Recipes for H2O Driverless AI\n")
for dirpath, dirs, files in os.walk("."):
    if all(x not in dirpath for x in exclude if len(x) > 1):
        path = dirpath.split('/')
        pdir = os.path.basename(dirpath)
        if pdir not in exclude:
            print(' ' * (len(path) - 2) + "* " + "[" + pdir + "](" + dirpath + ")")
            for f in files:
                if f not in exclude:
                    if f[-3:] == ".py":
                        print(' ' * (len(path) + 2) + "* [" + f + "](" + dirpath + "/" + f + ")")
