#### **Air-gapped Installation of Custom Recipes**

To use custom recipes on a air-gapped environment, do the following steps.

1) Download the required version of DriverlessAI TAR SH installer in a Internet facing machine from the following URL: 
    https://www.h2o.ai/download/

2) Run the following commands to install the DriverlessAI TAR SH. Replace VERSION with your specific version.

    chmod 755 dai-VERSION.sh
    ./dai-VERSION.sh
    
3) Now cd to the unpacked directory.

    cd dai-VERSION

3) Download the load_custom_recipe.py script from https://github.com/h2oai/driverlessai-recipes/air-gapped_installations

4) Run the following python script, either in one of the following ways:

    a) To load custom recipes from a local file.
        ./dai-env.sh python load_custom_recipe.py -username `<user>` -p `</absolute/path/to/the/custom_recipe/file.py>` >> load_custom_recipe.log
    
    b) To load custom recipes from a URL.
        ./dai-env.sh python load_custom_recipe.py -username `<user>` -u `<URL>` >> load_custom_recipe.log

    where `<user>` is the username, e.g. jon and `<URL>` is an http link for a url.
            
5) Once the above script was executed successfully, custom recipes and python dependencies will be installed in the  
        dai-VERSION/tmp/contrib directory.            
    
6) Zip the dai-VERSION/tmp/contrib directory, move it to the air-gapped machine and unzip there into the DAI tmp directory.
