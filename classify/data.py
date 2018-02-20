import json
import math

def create_sample(data):
	s = sorted(data, key=lambda d: d['label'])

	res = []
	cur_label = 0
	for cur_label in range(0, 5):
		found = False
		for f in s:
			if f['label'] == cur_label:
				res.append(f['direction'])
				res.append(f['length'])
				res.append(f['angle'])

				found = True
				break

		if not found:
			res.append(0)
			res.append(0)
			res.append(0)

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