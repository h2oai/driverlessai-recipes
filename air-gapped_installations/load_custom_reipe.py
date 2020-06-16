import argparse
import sys
from h2oaicore.recipe_server_support import server_load_all_custom_recipes


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
        server_load_all_custom_recipes(path=args.path)
    elif args.url:
        server_load_all_custom_recipes(url=args.url)
    else:
        print("Need to pass either -path or -url")


if __name__ == "__main__":
    main()