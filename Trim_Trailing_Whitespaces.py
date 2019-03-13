import glob

DEFAULT_CODE_FILE_EXT = [
    "*.cpp", "*.cc", "*.hpp",
    "*.c", "*.h",
    "*.py"
]


def deal(filename):
    text_data = []
    with open(filename, 'r', encoding='UTF-8') as fin:
        text_data = fin.readlines()
    text_data = [_.rstrip() for _ in text_data]
    while len(text_data[-1]) == 0:
        del text_data[-1]
    with open(filename, "w", encoding='UTF-8') as fout:
        for line in text_data:
            print(line, file=fout)
        # print(file=fout)
    print(filename, "done.")


def main():
    work_list = []
    for ext in DEFAULT_CODE_FILE_EXT:
        work_list += glob.glob(ext)
    for work in work_list:
        deal(work)


if __name__ == "__main__":
    main()
