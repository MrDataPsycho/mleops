import sys


python_version = ",".join(str(i) for i in sys.version_info[:])

if __name__ == "__main__":
    print("===")
    print(f"The python version is {python_version}")
    print("===")
