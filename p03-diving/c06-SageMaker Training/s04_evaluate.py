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


def glob_s3(glob_pattern):
    import boto3
    import fnmatch
    s3_client = boto3.client('s3')
    first_asterisk_index = glob_pattern.find('*')
    s3_prefix = glob_pattern[:first_asterisk_index] if first_asterisk_index >= 0 else glob_pattern
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    matched_objects = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        is_matched = fnmatch.fnmatch(key, glob_pattern)
        if is_matched:
            matched_objects.append(key)
    return matched_objects


def main():
    for filename in glob_s3("tmp/test"):
        print(predict(f"tmp/test/{filename}"))
