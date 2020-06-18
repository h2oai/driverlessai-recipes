import argparse
import os
from h2oaicore.recipe_server_support import server_load_all_custom_recipes

import re
regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def main():

    agr_parser = argparse.ArgumentParser(description='Load custom recipes',
                                         usage='./dai-env.sh python load_custom_recipe.py [options] -path or -url')

    args_group = agr_parser.add_mutually_exclusive_group(required=True)
    args_group.add_argument('-path',
                           type=str,
                           help='The path to custom recipes')

    args_group.add_argument('-url',
                            type=str,
                            help='The URL to custom recipes')

    args = agr_parser.parse_args()

    print(args.__dict__)
    if args.path:
        if not os.path.isfile(args.path):
            print("Not a valid file path")
            return
        server_load_all_custom_recipes(path=args.path)
    elif args.url:
        if not re.match(regex, args.url):
            print("Not a valid URL")
            return
        server_load_all_custom_recipes(url=args.url)
    else:
        print("Need to pass either -path or -url")


if __name__ == "__main__":
    main()