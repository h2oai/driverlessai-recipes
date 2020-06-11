import sys, getopt
from h2oaicore.recipe_server_support import server_load_all_custom_recipes


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hpu:", ["help", "recipe_path=", "recipe_url="])
        print(opts)
        if len(opts) == 0:
            raise getopt.GetoptError("")
    except getopt.GetoptError:
        print("Usage:")
        print('test.py -p <absolute/path/to/recipe>')
        print('test.py -u <url/to/recipe>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -p <absolute/path/to/recipe>')
            print('test.py -u <url/to/recipe>')
            sys.exit()
        elif opt in ("-p", "--recipe_path"):
            if len(arg) == 0:
                print("Not a valid path")
            else:
               server_load_all_custom_recipes(path=arg)
            sys.exit()
        elif opt in ("-u", "--recipe_url"):
            if len(arg) == 0:
                print("Not a valid URL")
            else:
               server_load_all_custom_recipes(url=arg)
            sys.exit()


if __name__ == "__main__":
    main(sys.argv[1:])