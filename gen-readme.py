import ast
exclude = ['.', '.idea', 'pycache', '.git', 'speech/data', 'Makefile', 'LICENSE', 'README.md', 'gen.sh', 'gen.py',
           '.pytest_cache', 'livecode']
sep = '  '


def get_module_docstring(filepath):
    co = compile(open(filepath).read(), filepath, 'exec')
    if co.co_consts and isinstance(co.co_consts[0], str):
        docstring = co.co_consts[0].replace("\n", "")
    else:
        docstring = None
    return docstring

def get_mojo_implementation_tag(filepath):
    with open(filepath, "r") as source:
        root = ast.parse(source.read())
        for node in (n for n in ast.walk(root) if isinstance(n, ast.Assign)):
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == '_mojo':
                if isinstance(node.value, ast.NameConstant) and node.value.value:
                    return ":hammer:"
        else:
            return ""
                    


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
                if f not in exclude and not f.startswith("test_"):
                    if f[-3:] == ".py":
                        docstring = get_module_docstring(os.path.join(dirpath, f)) or \
                                    "please add description"
                        what = "[" + f + "](" + dirpath + "/" + f + ")"
                        mojo_tag = get_mojo_implementation_tag(os.path.join(dirpath, f))
                        print_offset(depth + 1, "%s [%s] %s" % (what, docstring, mojo_tag), ret)
                        count += 1

print("""# Recipes for H2O Driverless AI

## About Driverless AI
H2O Driverless AI is Automatic Machine Learning for the Enterprise. Driverless AI automates feature engineering, model building, visualization and interpretability.
- Learn more about Driverless AI from the [H2O.ai website](https://www.h2o.ai/)
- Take the [test drive](https://www.h2o.ai/try-driverless-ai/)
- Go to the [Driverless AI community Slack channel](https://www.h2o.ai/community/driverless-ai-community/#chat) and ask your BYOR related questions in #general

## About BYOR
**BYOR** stands for **Bring Your Own Recipe** and is a key feature of Driverless AI. It allows domain scientists to solve their problems faster and with more precision.

## What are Custom Recipes?
Custom recipes are Python code snippets that can be uploaded into Driverless AI at runtime, like plugins. No need to restart Driverless AI. Custom recipes can be provided for transformers, models and scorers. During training of a supervised machine learning modeling pipeline (aka experiment), Driverless AI can then use these code snippets as building blocks, in combination with all built-in code pieces (or instead of). By providing your own custom recipes, you can gain control over the optimization choices that Driverless AI makes to best solve your machine learning problems.

## Best Practices for Recipes

### Security
* Recipes are meant to be built by people you trust and each recipe should be code-reviewed before going to production.
* Assume that a user with access to Driverless AI has access to the data inside that instance.
  * Apart from securing access to the instance via private networks, various methods of [authentication](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/authentication.html) are possible. Local authentication provides the most control over which users have access to Driverless AI.
  * Unless the `config.toml` setting `enable_dataset_downloading=false` is set, an authenticated user can download all imported datasets as .csv via direct APIs.
* When recipes are enabled (`enable_custom_recipes=true`, the default), be aware that:
  * The code for the recipes runs as the same native Linux user that runs the Driverless AI application.
    * Recipes have explicit access to all data passing through the transformer/model/scorer API
    * Recipes have implicit access to system resources such as disk, memory, CPUs, GPUs, network, etc.
  * A H2O-3 Java process is started in the background, for use by all recipes using H2O-3. Anyone with access to the Driverless AI instance can browse the file system, see models and data through the H2O-3 interface.

* Best ways to control access to Driverless AI and custom recipes:
  * Control access to the Driverless AI instance
  * Use local authentication to specify exactly which users are allowed to access Driverless AI
  * Run Driverless AI in a Docker container, as a certain user, with only certain ports exposed, and only certain mount points mapped
  * To disable all recipes: Set `enable_custom_recipes=false` in the config.toml, or add the environment variable `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES=0` at startup of Driverless AI. This will disable all custom transformers, models and scorers.
  * To disable new recipes: To keep all previously uploaded recipes enabled and disable the upload of any new recipes, set `enable_custom_recipes_upload=false` or `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES_UPLOAD=0` at startup of Driverless AI.

### Safety
* Driverless AI automatically performs basic acceptance tests for all custom recipes unless disabled
* More information in the FAQ

### Performance
* Use fast and efficient data manipulation tools like `data.table`, `sklearn`, `numpy` or `pandas` instead of Python lists, for-loops etc.
* Use disk sparingly, delete temporary files as soon as possible
* Use memory sparingly, delete objects when no longer needed

## Reference Guide
* [FAQ](https://github.com/h2oai/driverlessai-recipes/blob/master/FAQ.md#faq)
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
