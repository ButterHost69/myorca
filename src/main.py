from model import Model

def main():
    print("Hello from myorca!")
    my_model = Model()
    my_model.print_model()
    print(my_model.simple_infer("Hello, I am, "))


if __name__ == "__main__":
    main()
