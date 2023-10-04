import json

from IPython.display import Image, display


def get_class_from_results(results):
    results_prob_list = json.loads(results)
    best_index = results_prob_list.index(max(results_prob_list))

    return {
        0: "ZERO",
        1: "ONE",
        2: "TWO",
        3: "THREE",
        4: "FOUR",
        5: "FIVE",
        6: "SIX",
        7: "SEVEN",
        8: "EIGHT",
        9: "NINE"
    }[best_index]


def predict(filename, endpoint):
    byte_array_input = None

    with open(filename, 'rb') as image:
        f = image.read()
        byte_array_input = bytearray(f)

    display(Image(filename))

    results = endpoint.predict(byte_array_input)
    return get_class_from_results(results)


def main():
    for filename in glob.glob("tmp/test"):
        print(predict(f"tmp/test/{filename}"))
