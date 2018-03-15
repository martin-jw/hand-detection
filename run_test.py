import sys
import os
import re
import traceback

from classify import data, svc
import hand_detect as hd
import segmentation as seg

import multiprocessing
from multiprocessing import Pool
from functools import partial

import numpy as np
import csv

from time import time


def load_label_config(config_path):

    with open(config_path) as f:
        lines = f.readlines()

    config = {}
    pattern = re.compile('^(\d+)-(\d+) ?: ?(.+)')

    for line in lines:
        match = pattern.fullmatch(line.strip())

        if match is not None:
            for i in range(int(match.group(1)), int(match.group(2)) + 1):
                config[i] = match.group(3)
        else:
            print('Line', line, ' did not yield a match!')

    return config


def run_image(image_path, seg_method):
    drawing, finger_data, palm_point = hd.identify_image(seg_method, image_path)
    d = {}
    d['fingers'] = finger_data
    d['palm_point'] = palm_point

    proc_d, label = data.process_data(d)
    return data.create_sample(proc_d)


def test_image(image_path, config, classifier):
    d = ['-', '-', '-', '-', '-']
    path, name = os.path.split(image_path)
    match = re.search('id(\d+)', name)
    if match is not None:
        ind = int(match.group(1))
        d[0] = ind

        if ind in config:
            d[1] = config[ind]

        try:
            otsu = run_image(image_path, seg.create_bin_img_otsu)
            d[2] = classifier.predict([otsu])[0]
        except Exception as e:
            d[2] = 'FAILED'
            print('Otsu failed!')
            traceback.print_exc()

        try:
            slic = run_image(image_path, seg.create_bin_img_slic)
            d[3] = classifier.predict([slic])[0]
        except Exception as e:
            d[3] = 'FAILED'
            print('SLIC failed!')
            traceback.print_exc()

        try:
            ws = run_image(image_path, seg.create_bin_img_watershed)
            d[4] = classifier.predict([ws])[0]
        except Exception as e:
            d[4] = 'FAILED'
            print('Watershed failed!')
            traceback.print_exc()

    return d


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('Usage python', sys.argv[0], 'config-file training-dir in-dir out-dir')
    else:
        config_path = sys.argv[1]
        train_path = sys.argv[2]
        dir_path = sys.argv[3]
        outdir = sys.argv[4]

        config = load_label_config(config_path)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        files = []
        for f in os.listdir(train_path):
            if os.path.isfile(os.path.join(train_path, f)) and f.lower().endswith('.json'):
                files.append(os.path.join(train_path, f))

        X = np.empty(shape=(0, 15))
        y = np.empty(shape=(0))
        for file in files:
            sample_data, label = data.process_data(data.read_data_file(file))
            if not label == -1:
                X = np.append(X, [data.create_sample(sample_data)], axis=0)
                y = np.append(y, label)

        print("Training SVC with", X.shape[0], "samples!")

        classifier = svc.create_classifier(X, y)

        images = []
        for f in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, f)):
                images.append(os.path.join(dir_path, f))

        print("Testing dataset...")
        t0 = time()
        p = Pool(processes=multiprocessing.cpu_count() - 2)
        res = p.map(partial(test_image, classifier=classifier, config=config), images)

        print("Done in %0.3fs" % (time() - t0))
        print("Writing results...")
        with open(os.path.join(outdir, 'test.csv'), 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(res)


