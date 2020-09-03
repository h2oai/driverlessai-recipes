# Air-gapped Installation of Custom Recipes

Air gapping is a network security measure employed on one or more computers 
to ensure that a secure computer network is physically isolated from unsecured networks, 
such as the public Internet or an unsecured local area network. 

This documentation will guide you through the installation of Custom Recipes in such air-gapped environment.

## Prerequisite 
- There are two DAI installations involved here. One is in an **Air-Gapped Machine** and second is in an **Internet-Facing Machine**. They will be referred this way here to avoid confusion.
- First you need to install DAI in an air gapped environment *ie. in a computer isolated from internet*. 
This DAI can be installed in any Package Type (TAR SH, Docker, DEB etc.) available in https://www.h2o.ai/download/ 
- While in the Internet Facing Machine, clone this repository for now and checkout the branch of your DAI `VERSION`
```
  git clone https://github.com/h2oai/driverlessai-recipes.git
  cd driverlessai-recipes
  git checkout rel-VERSION # eg. git checkout rel-1.9.0 for DAI 1.9.0
```

## Installation Guide 
Follow along these steps to use custom recipes for DAI in an air-gapped environment:

*Note: following steps need to be performed in Internet Facing Machine*

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
    ./dai-env.sh python load_custom_recipe.py -username <user> -p `</absolute/path/to/the/custom_recipe/file.py>` >> load_custom_recipe.log
    ```
  where `<user>` is the username, e.g. jon and `</absolute/path/to/the/custom_recipe/file.py>` is the path to a recipe you want to upload to DAI. 
  
  For example to load [daal_trees recipe](https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/daal_trees.py) from the cloned driverlessai-recipes repo we do:
    ```
  ./dai-env.sh python load_custom_recipe.py -username jon -p /home/ubuntu/driverlessai-recipes/models/algorithms/daal_trees.py >> load_custom_recipe.log
    ```

  b) To load custom recipes from a URL.
    ```
    ./dai-env.sh python load_custom_recipe.py -username <user> -u <URL> >> load_custom_recipe.logg
    ```
  where `<URL>` is an http link for a url.
  
  For example to load [catboost recipe](https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/catboost.py) from url we do:
    ```
    ./dai-env.sh python load_custom_recipe.py -username jon -u https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/daal_trees.py >> load_custom_recipe.log
    ```
  **Note:** you can check the `load_custom_recipe.log` file to see if the operation was successful.
            
- Once the above script was executed successfully, custom recipes and python dependencies will be installed in the  
        `dai-VERSION/<data_directory>/<user>/contrib` directory, 
        where `<data_directory>` is `tmp` by default.           
    
- Zip the `dai-VERSION/tmp/contrib` directory and move it to the air-gapped machine and unzip there into the DAI `tmp` directory.
```
  cd dai-VERSION/`<data_directory>`/
  zip -r user_contrib.zip `<user>`/contrib
  scp user_contrib.zip `<remote_user>`@`<remote_system>`:`<path to data_directory on remote system>`
```
- Now in the **Air-gapped Machine**, unzip the file and set permissions if necessary, e.g.
```
  cd `<dai data_directory>`
  unzip user_contrib.zip
  chmod -R u+rwx dai:dai `<user>`/contrib
```