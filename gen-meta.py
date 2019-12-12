# -*- coding: utf-8 -*-

import os

exclude = ['.', '.idea', 'pycache', '.git', 'Makefile', 'LICENSE', 'README.md', 'gen.sh', 'gen.py', '.pytest_cache', 'www']

def get_module_docstring(filepath):
    co = compile(open(filepath).read(), filepath, 'exec')
    if co.co_consts and isinstance(co.co_consts[0], str):
        docstring = co.co_consts[0].replace("\n", "")
    else:
        docstring = None
    return docstring

def gen_meta_yaml(
        category,
        branch="master",
        baseurl="https://github.com/h2oai/driverlessai-recipes/blob/{}"
        ):
    for dirpath, dirs, files in os.walk(category):
        baseurl = baseurl.format(branch)
        if all(x not in dirpath for x in exclude if len(x) > 1):
            path = dirpath.split('/')
            category = path[0]
            tags = path[1:]
            for f in sorted(files):
                if f not in exclude and not f.startswith("test_"):
                    if f[-3:] == ".py":
                        name = f[:-3].replace("_", " ").replace("-", " ").title()
                        docstring = get_module_docstring(os.path.join(dirpath, f)) or "🙄 Not available yet ..."
                        url = "{}/{}".format(baseurl, os.path.join(dirpath, f))
                        print(
                                f"- recipe:\n"
                                f"   category: '{category}'\n"
                                f"   name: '{name}'\n"
                                f"   desc: >\n"
                                f"         {docstring}\n"
                                f"   url: '{url}'\n"
                                f"   tags: {','.join(tags)}\n"
                            )


def main():
    gen_meta_yaml("models")
    gen_meta_yaml("scorers")
    gen_meta_yaml("transformers")
    gen_meta_yaml("data")

if __name__ == '__main__':
    main()

