#!/bin/bash
tree=$(tree -f --noreport -I '*~' --charset ascii $1 |
       sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g' | grep -v '.pyc' | grep -v 'LICENSE' | grep -v 'Makefile' | grep -v 'README.md' | grep -v 'gen.sh' | grep -v '^.$')
printf "# Recipes for H2O Driverless AI\n\n${tree}" > README.md
