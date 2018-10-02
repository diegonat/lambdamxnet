import ctypes
import os
from six.moves import urllib
import zipfile
import stat
import logging
import json
import numpy as np
from one_hot import one_hot


logging.basicConfig(level=logging.DEBUG)
print "logging"

def response(status_code, response_body):
    return {
                'statusCode': status_code,
                'body': str(response_body),
                'headers': {
                    'Content-Type': 'application/json',
                },
            }

def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1.
    return results

def one_hot_encode(messages, dimension):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_lenght)
        data.append(temp)
    return data

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a') or f.endswith('.settings'):
            continue
        print('loading %s...' % f)
        ctypes.cdll.LoadLibrary(os.path.join(d, f))



path_to_model = "./"
vocabulary_lenght = 9013


import mxnet as mx

model_dir = "./"
vocabulary_lenght = 9013


symbol = mx.sym.load('%s/model.json' % model_dir)
outputs = mx.symbol.softmax(data=symbol, name='softmax_label')
inputs = mx.sym.var('data')
param_dict = mx.gluon.ParameterDict('model_')
net = mx.gluon.SymbolBlock(outputs, inputs, param_dict)
net.load_params('%s/model.params' % model_dir, ctx=mx.cpu())


def handler(event, context):
    print event
    #params = event['queryStringParameters']

    sms = event['body']

    print sms

    test_messages = [sms.encode('ascii','ignore')]


    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_lenght)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_lenght)

    encoded_test_messages = mx.nd.array(encoded_test_messages)
    output = net(encoded_test_messages)
    predictions = np.argmax(output, axis=1)

    print(predictions[0])

    result = predictions[0]

    return response(200, result)
