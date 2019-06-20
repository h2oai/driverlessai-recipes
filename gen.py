exclude = ['.', '.git', 'data', 'Makefile', 'LICENSE', 'README.md', 'gen.sh', 'gen.py']
sep = '  '

def get_module_docstring(filepath):
    co = compile(open(filepath).read(), filepath, 'exec')
    if co.co_consts and isinstance(co.co_consts[0], str):
        docstring = co.co_consts[0]
    else:
        docstring = None
    return docstring

def print_offset(depth, str_content):
    for i, line in enumerate(str_content.split("\n")):
        if i == 0:
            print(sep * depth + "* " + line)
        else:
            print(sep * depth + "  " + line)

import os
print("# Recipes for H2O Driverless AI\n")
for dirpath, dirs, files in os.walk("."):
    if all(x not in dirpath for x in exclude if len(x) > 1):
        path = dirpath.split('/')
        pdir = os.path.basename(dirpath)
        if pdir not in exclude:
            depth = len(path) - 2
            # print(sep * depth + "* " + "[" + pdir + "](" + dirpath + ")")
            print_offset(depth, "[" + pdir + "](" + dirpath + ")")
            for f in files:
                if f not in exclude:
                    if f[-3:] == ".py":
                        docstring = get_module_docstring(os.path.join(dirpath, f)) or "please add documentation"
                        what = "[" + f + "](" + dirpath + "/" + f + ")"
                        print_offset(depth + 1, "/**\n%s\n*/\n%s" % (docstring, what))
