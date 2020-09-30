import json

path = 'real_export__pos4/data.json'

with open(path, 'r') as fp:
    data = json.load(fp)

i = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
while True:
    ky = 'run_%d' % i
    if ky not in data:
        break
    if data[ky]['result'] == 'target':
        cnt1 += 1
    if data[ky]['result'] == 'antitarget':
        cnt2 += 1
    if data[ky]['result'] == 'interrupt':
        cnt3 += 1
    if data[ky]['result'] == 'timeout':
        cnt4 += 1
    i += 1

    print(data[ky]['result'])

print('%d of %d target' % (cnt1, i))
print('%d of %d antitarget' % (cnt2, i))
print('%d of %d interrupt' % (cnt3, i))
print('%d of %d timeout' % (cnt4, i))


#
# import json
#
# path = 'data/meta/position3.json'
#
# with open(path, 'r') as fp:
#     data = json.load(fp)
#
# i = 0
# cnt = 0
# while True:
#     ky = 'step_%d' % i
#     if ky not in data:
#         break
#     i += 1
# input(i)
# for j in range(i - (21 * 24), i):
#     del data['step_%d' % j ]
#
# with open('data/meta/20190731ssss.json', 'w') as fp:
#     json.dump(data, fp)


#
# import json
#
# path = 'data/meta/position4.json'
#
# with open(path, 'r') as fp:
#     data = json.load(fp)
#
# sm = data['trackers']['red_small']
# bg = data['trackers']['red_big']
#
# data['trackers']['red_small'] = bg
# data['trackers']['red_big'] = sm
#
# with open('data/meta/position4.json', 'w') as fp:
#     json.dump(data, fp)
