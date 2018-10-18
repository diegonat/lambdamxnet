import ctypes
import os
import logging
import numpy as np
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

logging.basicConfig(level=logging.DEBUG)
print "logging"

def response(status_code, response_body):
    return {
                'statusCode': status_code,
                'body': str(response_body),
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin' : '*',
                    'Access-Control-Allow-Credentials' : 'true',
		    'Access-Control-Allow-Headers': '*'
                },
            }

for d, _, files in os.walk('lib'):
    for f in files:
        if f.endswith('.a') or f.endswith('.settings'):
            continue
        print('loading %s...' % f)
        ctypes.cdll.LoadLibrary(os.path.join(d, f))

import mxnet as mx

vocabulary_lenght = 9013

# Load the Gluon model.
net = mx.gluon.nn.SymbolBlock(
        outputs=mx.sym.load('./model.json'),
        inputs=mx.sym.var('data'))
net.load_params('./model.params', ctx=mx.cpu())


def handler(event, context):

    sms = event['body']


    if 'httpMethod' in event:
        if event['httpMethod'] == 'OPTIONS':
            return response(200, '')

        elif event['httpMethod'] == 'POST':
            test_messages = [sms.encode('ascii','ignore')]

            one_hot_test_messages = one_hot_encode(test_messages, vocabulary_lenght)
            encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_lenght)

            encoded_test_messages = mx.nd.array(encoded_test_messages)
            output = net(encoded_test_messages)
            sigmoid_output = output.sigmoid()
            prediction = mx.nd.abs(mx.nd.ceil(sigmoid_output - 0.5))
            
            output_obj = {}
            output_obj['predicted_label'] = np.array2string(prediction.asnumpy()[0][0])
            output_obj['predicted_probability'] = np.array2string(sigmoid_output.asnumpy()[0][0])

            return response(200, output_obj)

        else:
            return response(405, 'null')
