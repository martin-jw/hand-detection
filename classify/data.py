import json
import math


def create_sample(data):
    s = sorted(data, key=lambda d: d['point'][1])

    if s[0]['label'] != "thumb":
        s = sorted(data, key=lambda d: d['point'][1], reverse=True)

    res = []
    cur_label = 0

    fingers = [False, False, False, False, False]

    if len(s) > 5:
        print("Warning: sample has more than 5 fingers!")

    for cur_label in range(0, 5):
        found = False
        for f in s:
            if f['label'] == cur_label:
                res.append(f['direction'])
                res.append(f['length'])
                res.append(f['angle'])

                found = True
                s.remove(f)
                break

        if not found:
            res.append(0)
            res.append(0)
            res.append(0)

    for i in range(3, 15, 3):
        if res[i + 1] == 0 and len(s) > 0:
            f = s[0]
            res[i] = f['direction']
            res[i + 1] = f['length']
            res[i + 2] = f['angle']

    return res


def process_data(data):
    result = []

    pp = data['palm_point']

    for finger in data['fingers']:
        f = {}

        label_ids = {
            "thumb": 0,
            "index": 1,
            "middle": 2,
            "ring": 3,
            "pinky": 4
        }

        tip = finger['tip']
        start = finger['start']

        direction = math.degrees(math.atan2(-1*(tip[0] - start[0]), (tip[1] - start[1])))
        f['direction'] = direction
        f['point'] = finger['point']
        f['label'] = label_ids[finger['label']]
        f['angle'] = finger['angle_from_pp']
        f['length'] = finger['length']

        result.append(f)

    if 'label' in data:
        return result, data['label']

    return result, -1


def read_data_file(data_path):
    print("Reading file:", data_path)
    with open(data_path) as data:
        d = json.load(data)
        return d


if __name__ == "__main__":
    print(process_data(read_data_file('data.json')))