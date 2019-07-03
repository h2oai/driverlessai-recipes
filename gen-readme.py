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

print("""# Recipes for H2O Driverless AI

## About Driverless AI
H2O Driverless AI is Automatic Machine Learning for the Enterprise. Driverless AI automates feature engineering, model building, visualization and interpretability.
- Learn more about Driverless AI from the [H2O.ai website](https://www.h2o.ai/)
- Take the [test drive](https://www.h2o.ai/try-driverless-ai/)
- Go to the [Driverless AI community Slack channel](https://www.h2o.ai/community/driverless-ai-community/#chat) and ask your BYOR related questions in #general

## About BYOR
**BYOR** stands for **Bring Your Own Recipe** and is a key feature of Driverless AI. It allows domain scientists to solve their problems faster and with more precision.

## Reference Guide
* [FAQ](https://github.com/h2oai/driverlessai-recipes/blob/master/FAQ.md#faq)
* [Templates](https://github.com/h2oai/driverlessai-recipes/blob/master/FAQ.md#references)
* [Technical Architecture Diagram](https://raw.githubusercontent.com/h2oai/driverlessai-recipes/master/reference/DriverlessAI_BYOR.png)
""")
print("## Sample Recipes")
print("### Count: %d" % count)
for l in ret:
    print(l)
