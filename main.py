# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from common.get_good_model import get_good_model
from evaluate.evaluate import evaluate_model

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('start')
    dims = [30]
    evaluate_model(dims=dims, processes=24, models=get_good_model('data/good_model_t100.txt'), runtimes=20,
                   prob_1=0.5)
    evaluate_model(dims=dims, processes=24, models=get_good_model('data/good_model_t100.txt'), runtimes=100,
                   prob_1=0.4)
    evaluate_model(dims=dims, processes=24, models=get_good_model('data/good_model_t100.txt'), runtimes=100,
                   prob_1=0.3)
    print('end')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
