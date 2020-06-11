#### **Air-gapped Installation of Custom Recipes**

To use install custom recipes on a air-gapped environment, do the following steps.

1) Download the required version of Driverless AI TAR SH installer from the following URL: https://www.h2o.ai/download/

2) Run the following commands to install the Driverless AI RPM. Replace VERSION with your specific version.

    chmod 755 dai-VERSION.sh
    ./dai-VERSION.sh
    
3) Now cd to the unpacked directory.

    cd dai-VERSION

3) Download the load_custom_recipe.py script from https://github.com/h2oai/driverlessai-recipes/air-gapped_installations

4) Run the following python script, either in one of the following ways:

    a) To load custom recipes from a local file.
            ./dai-env.sh python load_custom_recipe.py -p </absolute/path/to/the/custom_recipe/file.py>
    
    b) To load custom recipes from a URL.
            ./dai-env.sh python load_custom_recipe.py -u <URL>
            
5) Once the above script was executed successfully, custom recipes and python dependencies will be
    installed in the  dai-VERSION/tmp/contrib directory.
    
6) Zip the dai-VERSION/tmp/contrib directory, move it to the air-gapped machine and unzip there.                  

        