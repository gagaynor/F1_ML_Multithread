import os


def main():
    cpu_count = os.cpu_count()

    print (f'Number of cpus in system: {cpu_count}')

if __name__ == "__main__":
    main()