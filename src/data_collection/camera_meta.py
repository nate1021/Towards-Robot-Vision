import os
import json


def _load_relevent_camera_jsons():
    arr = []
    DIR = "../../data/raw/camera/meta"  # manually copy correct json files to this folder
    fi = 1
    while True:
        pth = os.path.join(DIR, 'position%d.json' % fi)
        if 7 <= fi <= 9:
            pass
        elif os.path.isfile(pth):
            with open(pth, 'r') as fp:
                jsn = json.load(fp)
                arr.append(jsn)
        else:
            break
        fi += 1
    return arr


def load_trackers():
    trackers_arr = []
    arr = _load_relevent_camera_jsons()
    fi = 1
    for jsn in arr:
        trackers = jsn['trackers']
        trackers['position'] = fi
        trackers_arr.append(trackers)
        fi += 1
    return trackers_arr


def load_json(index):
    return _load_relevent_camera_jsons()[index]
