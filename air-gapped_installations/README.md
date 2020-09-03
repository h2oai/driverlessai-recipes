# **Air-gapped Installation of Custom Recipes**

Air gapping is a network security measure employed on one or more computers 
to ensure that a secure computer network is physically isolated from unsecured networks, 
such as the public Internet or an unsecured local area network. 

This documentation will guide you through the installation of Custom Recipes in such air-gapped environment.

## Prerequisite 
- There are two DAI installations involved here. One is an **Air-Gapped DAI** and second is an **Internet-Facing DAI**. They will be referred this way in this doc to avoid confusion.
- First you need to install DAI in an air gapped environment *ie. in a computer isolated from internet*. 
This DAI can be installed in any Package Type (TAR SH, Docker, DEB etc.) available in https://www.h2o.ai/download/ 
- While in the Internet Facing DAI, clone this repository for now and checkout the branch of your DAI `VERSION`
```
  git clone https://github.com/h2oai/driverlessai-recipes.git
  cd driverlessai-recipes
  git checkout rel-VERSION # eg. git checkout rel-1.9.0 for DAI 1.9.0
```

## Installation Guide 
Follow along these steps to use custom recipes for DAI in an air-gapped environment:

*Note: following steps need to be performed in Internet Facing DAI*

- Download the required version of Driverless AI TAR SH installer in a Internet facing machine from https://www.h2o.ai/download/

- Run the following commands to install the Driverless AI TAR SH. Replace `VERSION` with your specific version.

```
  chmod 755 dai-VERSION.sh
  ./dai-VERSION.sh
```
- Now cd to the unpacked directory.
```
  cd dai-VERSION
```
- Copy the load_custom_recipe.py script from `driverlessai-recipes/air-gapped_installations` to `dai-VERSION`

- Run the following python script, either in one of the following ways:

  a) To load custom recipes from a local file.
    ```
    ./dai-env.sh python load_custom_recipe.py -p </absolute/path/to/the/custom_recipe/file.py> >> load_custom_recipe.log
    ```
  `</absolute/path/to/the/custom_recipe/file.py>` = path to a recipe you want to upload to DAI. 
  
  For example to load [daal_trees recipe](https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/daal_trees.py) from the cloned driverlessai-recipes repo we do:
    ```
  ./dai-env.sh python load_custom_recipe.py -p /home/ubuntu/driverlessai-recipes/models/algorithms/daal_trees.py >> load_custom_recipe.log
    ```

  b) To load custom recipes from a URL.
    ```
    ./dai-env.sh python load_custom_recipe.py -u <URL> >> load_custom_recipe.log
    ```
  where `<URL>` is an http link for a url.
  For example to load [catboost recipe](https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/catboost.py) from url we do:
    ```
    ./dai-env.sh python load_custom_recipe.py -u https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/daal_trees.py >> load_custom_recipe.log
    ```
  **Note:** you can check if the operation was successful by checking the `load_custom_recipe.log` file. 
            
- Once the above script was executed successfully, custom recipes and python dependencies will be installed in the  
        `dai-VERSION/tmp/contrib` directory.            
    
- Zip the `dai-VERSION/tmp/contrib` directory and move it to the air-gapped machine and unzip there into the DAI `tmp` directory.
```
cd dai-VERSION/tmp/
zip -r contrib.zip contrib
scp contrib.zip <remote_user>@<remote_system>:<path to data_directory on remote system>
```
- Now in the **Air-gapped** machine, unzip the file and set permissions if necessary, e.g.
```
cd <dai data_directory>
unzip contrib.zip
chmod -R u+rwx contrib
```