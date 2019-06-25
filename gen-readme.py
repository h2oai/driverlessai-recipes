exclude = ['.', '.idea', 'pycache', '.git', 'data', 'Makefile', 'LICENSE', 'README.md', 'gen.sh', 'gen.py']
sep = '  '


def get_module_docstring(filepath):
    co = compile(open(filepath).read(), filepath, 'exec')
    if co.co_consts and isinstance(co.co_consts[0], str):
        docstring = co.co_consts[0].replace("\n", "")
    else:
        docstring = None
    return docstring


def print_offset(depth, str_content, ret):
    for i, line in enumerate(str_content.split("\n")):
        if i == 0:
            ret.append(sep * depth + "* " + line)
        else:
            ret.append(sep * depth + "  " + line)


import os

count = 0
ret = []
for dirpath, dirs, files in os.walk("."):
    dirs.sort()
    if all(x not in dirpath for x in exclude if len(x) > 1):
        path = dirpath.split('/')
        pdir = os.path.basename(dirpath)
        if pdir not in exclude:
            depth = len(path) - 2
            print_offset(depth, "[" + pdir.upper() + "](" + dirpath + ")", ret)
            for f in sorted(files):
                if f not in exclude:
                    if f[-3:] == ".py":
                        docstring = get_module_docstring(os.path.join(dirpath, f)) or \
                                    "please add description"
                        what = "[" + f + "](" + dirpath + "/" + f + ")"
                        print_offset(depth + 1, "%s [%s]" % (what, docstring), ret)
                        count += 1

print("# Recipes for H2O Driverless AI\n")
print("## Reference Guide")
print("* [FAQ and Templates](./FAQ.md)")
print("* [Technical Architecture Diagram](https://raw.githubusercontent.com/h2oai/driverlessai-recipes/master/reference/DriverlessAI_BYOR.png)")
print("")
print("## Sample Recipes: %d" % count)
for l in ret:
  print(l)
