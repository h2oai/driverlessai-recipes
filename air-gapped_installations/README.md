# Air-gapped Installation of Custom Recipes

Air gapping is a network security measure used to isolate one or more computers from unsecured networks, such as the public Internet or an unsecured local area network. This ensures that the secure computer network remains physically separated from external, potentially unsafe networks.

This guide walks you through the installation process of Custom Recipes in an air-gapped environment.

## Prerequisite 
- Two DAI installations are required: one on an **Air-Gapped Machine** and the other on an **Internet-Facing Machine**. These terms will be used throughout the document for clarity.
- First, install DAI on the air-gapped machine (a machine isolated from the Internet). DAI can be installed using any available package type (e.g., TAR SH, Docker, DEB). You can download the installation packages from [H2O.ai Downloads](https://h2o.ai/resources/download/).
- On the Internet-facing machine, clone the repository and check out the appropriate branch for your DAI `VERSION`
```
  git clone https://github.com/h2oai/driverlessai-recipes.git
  cd driverlessai-recipes
  git checkout rel-VERSION # eg. git checkout rel-1.9.0 for DAI 1.9.0
```

## Installation Guide
Follow the steps below to use custom recipes for DAI in an air-gapped environment:

**Note**: *These steps should be performed on the Internet-Facing Machine.*

1. Download the required version of the Driverless AI TAR SH installer on the Internet-Facing Machine from [H2O.ai Downloads](https://www.h2o.ai/download/).

2. Run the following commands to install the Driverless AI TAR SH. Replace `VERSION` with the specific version you need.
  ```
    chmod 755 dai-VERSION.sh
    ./dai-VERSION.sh
  ```

3. Next, `cd` to the unpacked directory:
  ```
    cd dai-VERSION
  ```

4. Copy the `load_custom_recipe.py` script from `driverlessai-recipes/air-gapped_installations` to `dai-VERSION`.

5. Run the following Python script, either in one of the following ways:

  - **To load custom recipes from a local file:**
    ```
    ./dai-env.sh python load_custom_recipe.py -username <user> -p `</absolute/path/to/the/custom_recipe/file.py>` >> load_custom_recipe.log
    ```
     - `<user>`: The username (e.g., `jon`).
     - `</absolute/path/to/the/custom_recipe/file.py>`: The path to the recipe you want to upload to DAI.
  
   For example, to load the [daal_trees recipe](https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/daal_trees.py) from the cloned `driverlessai-recipes` repo:
   ```
   ./dai-env.sh python load_custom_recipe.py -username jon -p /home/ubuntu/driverlessai-recipes/models/algorithms/daal_trees.py >> load_custom_recipe.log
   ```

  - **To load custom recipes from a URL:**
    ```
    ./dai-env.sh python load_custom_recipe.py -username <user> -u <URL> >> load_custom_recipe.logg
    ```
      - `<URL>`: The URL to the custom recipe.
  
   For example, to load the [catboost recipe](https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/catboost.py) from a URL:
   ```
   ./dai-env.sh python load_custom_recipe.py -username jon -u https://github.com/h2oai/driverlessai-recipes/blob/rel-1.8.8/models/algorithms/catboost.py >> load_custom_recipe.log
   ```
  **Note:** *You can check the `load_custom_recipe.log` file to verify if the operation was successful.*
            
6.Once the script has been executed successfully, custom recipes and Python dependencies will be installed in the `dai-VERSION/<data_directory>/<user>/contrib` directory,  where `<data_directory>` is `tmp` by default.   
    
7. Zip the `dai-VERSION/tmp/contrib` directory and move it to the air-gapped machine. Unzip it into the DAI `tmp` directory:
  ```
    cd dai-VERSION/`<data_directory>`/
    zip -r user_contrib.zip `<user>`/contrib
    scp user_contrib.zip `<remote_user>`@`<remote_system>`:`<path to data_directory on remote system>`
  ```

8. Now, on the **Air-Gapped Machine**, unzip the file and set permissions if necessary:
   ```
   cd <dai data_directory>
   unzip user_contrib.zip
   chmod -R u+rwx dai:dai <user>/contrib
   ```
