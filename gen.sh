#!/bin/bash
tree=$(tree -tf --noreport -I '*~' --charset ascii $1 |
       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g' | grep -v '.pyc')
printf "# Recipes for H2O Driverless AI\n\n${tree}" > README.md
