from random import randint

with open('../data/synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

true_answers = {}
with open('keras_resnet50.csv') as f:
    for line in f:
        x = line.split(',')
        true_answers[x[2]] = x[3]

w = open('32predict.csv', 'w')
count = 0
with open('/Users/danilrudenko/rails_test/Danone-11d-prototype/paper/32predictions.txt') as f:
    for line in f:
        x = line.split(',')
        true_path = '..' + x[0][18:]
        predict_class = int(x[1])
        predict_value = 0
        if predict_class > 999 or predict_class < 0:
            predict_class = randint(0, 999)
        else:
            predict_value = float(x[2])
        predict_time = float(x[3]) / 1000
        s = ','.join(
            map(str,
                [labels[predict_class].split(' ')[0], predict_value, true_path, true_answers[true_path], 'tensrort',
                 predict_time, 'vgg16']))
        w.write(s)
        w.write('\n')
        count += 1

print(count)
w.close()
