exclude = ['.idea', 'pycache', '.git', 'speech/data', 'Makefile', 'LICENSE', 'README.md', 'gen.sh', 'gen.py',
           '.pytest_cache', 'gen-readme.py']
sep = '  '


def get_file_docstring(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    docstring = ""
    for line in lines:
        first_char = line[0]
        if not first_char == '#':
            break
        else:
            docstring += line[1:]
    return docstring

def print_offset(depth, str_content, ret):
    for i, line in enumerate(str_content.split("\n")):
        if i == 0:
            ret.append(sep * depth + "* " + line)
        else:
            ret.append(sep * depth + "  " + line)

import os
from os import listdir
from os.path import isfile, join

count = 0
ret = []
livecode_path = "./data/livecode"
livecode_files = [f for f in listdir(livecode_path) if isfile(join(livecode_path, f))]
for f in sorted(livecode_files):
            depth = 0
            if f not in exclude and not f.startswith("test_"):
                if f[-3:] == ".py":
                        docstring = get_file_docstring(os.path.join(livecode_path, f)) or \
                                    "please add description"
                        what = "[" + f + "](" + livecode_path + "/" + f + ")"
                        print_offset(depth + 1, "%s [%s]" % (what, docstring), ret)
                        count += 1

print("""# Recipes for H2O Driverless AI

## About Driverless AI and BYOR 
See main [README](https://github.com/h2oai/driverlessai-recipes/README.md)

## About Data Recipes
Driverless AI allows you to create a new dataset by modifying an existing dataset with a data recipe. 
When inside **Dataset Details** page click the **Modify by Recipe** button in the top right portion of the UI, then click **Live Code** from the submenu that appears.
The Dialog Box that appears will feature an editor to enter the code for the data recipe you want to use to modify the dataset. The list of templates below are designed
for the live code applications with certain modifications specific to the dataset they apply to.

## Reference Guide
* [Adding a Data Recipe](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/custom-recipes-data-recipes.html#adding-a-data-recipe)
* [Templates](https://github.com/h2oai/driverlessai-recipes/blob/master/FAQ.md#references)
* [Technical Architecture Diagram](https://raw.githubusercontent.com/h2oai/driverlessai-recipes/master/reference/DriverlessAI_BYOR.png)
""")
print("## Sample Recipes")
print("[Go to Recipes for Driverless 1.7.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.7.0)")
print(" [1.7.1](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.7.1)")
print(" [1.8.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.0)")
print(" [1.8.1](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.1)")
print(" [1.8.2](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.2)")
print(" [1.8.3](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.3)")
print(" [1.8.4](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.4)")
print(" [1.8.5](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.5)")
print(" [1.8.6](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.6)")
print(" [1.8.7](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.7)")
print(" [1.8.8](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.8)")
print(" [1.9.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.9.0)")
print("### Count: %d" % count)
for l in ret:
    print(l)
