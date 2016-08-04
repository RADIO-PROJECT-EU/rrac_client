#!/usr/bin/env python2.7

import argparse
import glob
import json
import logging
import os
import shutil
import time
import collections
import sys
import signal
import wave

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import numpy as np
import paho.mqtt.client as mqtt

from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioFeatureExtraction as aF

from utils import *

ROOT_TOPIC = '/rrac'
ROOT_FOLDER = os.environ.get('RRAC_HOME', os.path.dirname(os.path.realpath(__file__)))
AUDIO_DEVICE = None

GST_PIPELINE = None
MQTT_CLIENT = None
CLASSIFIER = None
RECORDINGS = None

EVENT_CLASSIFIER = {
    'OBJ': None,
    'MEAN': None,
    'STD': None,
    'CLASSES': None
}

# -------------------------------------
#          Helper functions
# -------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description = 'RRAC client options',\
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device',   default='default',              help='Input audio device')
    parser.add_argument('--url',      default='143.233.226.172:1883', help='URL to the MQTT broker')
    parser.add_argument('--username', default=None,                   help='Username for MQTT connection')
    parser.add_argument('--password', default=None,                   help='Password for MQTT connection')
    return parser.parse_args()

def populate_recordings():
    recordings = collections.defaultdict(int)
    events = next(os.walk(os.path.join(ROOT_FOLDER, 'recordings')))[1]

    for event in events:
        logging.debug('found recordings folder for event {}'.format(event))
        folder = os.path.join(ROOT_FOLDER, 'recordings', event)
        for recording in glob.glob(os.path.join(folder, '*.wav')):
            recordings[event] += wav_duration(recording)

    return recordings
    #return [{'name': e, 'duration': recordings[e]} for e in recordings]

def populate_classifiers():
    # A classifier consists of 3 files: xxxx, xxxx.arff, xxxxMEANS
    classifiers = []
    arffs = glob.glob(os.path.join(ROOT_FOLDER, 'classifiers', '*.arff'))
    names = [os.path.splitext(x)[0] for x in arffs]

    for n in names:
        _, _, _, classes, _, _, _, _, _ = aT.loadSVModel(os.path.join(ROOT_FOLDER, 'classifiers', n))
        classifiers.append({'name': os.path.basename(n), 'events': classes})

    return classifiers

# ----------------------------------------------
#      GStreamer/pyAudioAnalysis functions
# ----------------------------------------------
def recording_pipeline():
    l = ' ! '.join([
        'alsasrc device={}'.format(AUDIO_DEVICE),
        'audioconvert',
        'audioresample',
        'audio/x-raw, rate=16000, channels=1',
        'wavenc',
        'filesink name=sink'
    ])

    return Gst.parse_launch(l)

def classification_pipeline():


    l = 'alsasrc device={} blocksize={} ! audioconvert ! audioresample ! audio/x-raw, rate=16000, channels=1, format=S16LE ! tee name=t\
     t. ! queue ! wavenc ! filesink location=runtimetest.wav\
     t. ! appsink name=sink emit-signals=true'.format(AUDIO_DEVICE, 16000)
    '''
    l = ' ! '.join([
        'alsasrc device={} blocksize={}'.format(AUDIO_DEVICE, 32000),
        'audioconvert',
        'audioresample',
        'audio/x-raw, rate=16000, channels=1, format=S16LE',
        'appsink name=sink emit-signals=true'
    ])
    '''
    return Gst.parse_launch(l)


def start_recording(event):
    global GST_PIPELINE

    folder = os.path.join(ROOT_FOLDER, 'recordings', event)
    logging.debug('recording directory is {}'.format(folder))
    if not os.path.exists(folder):
        os.makedirs(folder)

    recordings = glob.glob(os.path.join(folder, 'rec*.wav'))
    logging.debug('found recordings {}'.format(recordings))
    filename = os.path.join(folder, 'rec{0:04d}.wav'.format(len(recordings) + 1))

    logging.info('recording in {}'.format(filename))
    if GST_PIPELINE:
        GST_PIPELINE.set_state(Gst.State.NULL)
    GST_PIPELINE = recording_pipeline()
    GST_PIPELINE.get_by_name('sink').set_property('location', filename)
    GST_PIPELINE.set_state(Gst.State.PLAYING)

    return (True, filename)

def stop_recording():
    global GST_PIPELINE, RECORDINGS
    try:
        assert(GST_PIPELINE is not None)
        assert(GST_PIPELINE.get_state(0)[1] == Gst.State.PLAYING)
        sink = GST_PIPELINE.get_by_name('sink')
        assert(isinstance(sink, GstFileSink))
    except AssertionError:
        return (False, 'Not recording currently')

    status = GST_PIPELINE.set_state(Gst.State.NULL) == Gst.StateChangeReturn.SUCCESS
    if status:
        recording = sink.get_property('location')
        logging.info('fixing WAVE header of {}'.format(recording))
        fix_wav_header(recording)
        event = os.path.split(os.path.dirname(recording))[1]
        logging.info('stopped recording of event {}'.format(event))
        RECORDINGS[event] += wav_duration(recording)
        print('now recordings are {}'.format(dict(RECORDINGS)))
        return (True, RECORDINGS)
    else:
        return (False, 'Could not stop recording')

def classification_callback(sink):
    global MQTT_CLIENT, EVENT_CLASSIFIER

    gst_sample = sink.emit('pull-sample')
    gst_buffer = gst_sample.get_buffer()
    raw_data = gst_buffer.extract_dup(0, gst_buffer.get_size())
    data = np.frombuffer(raw_data, dtype=np.int16)
    features = aF.mtFeatureExtraction(data, 16000, 16000, 16000, aT.shortTermWindow * 16000, aT.shortTermStep * 16000)[0] # 1s mt / 20ms st windows
    features = features.mean(axis=1)
    normalized_features = (features - EVENT_CLASSIFIER['MEAN']) / EVENT_CLASSIFIER['STD']
    [r, p] = aT.classifierWrapper(EVENT_CLASSIFIER['OBJ'], 'svm', normalized_features)

    idx = int(r)
    event_notification = {
        't':      int(time.time()),
        'energy': features[1],
        'event':  EVENT_CLASSIFIER['CLASSES'][idx],
        'prob':   max(p)
    }

    MQTT_CLIENT.publish('{}/events'.format(ROOT_TOPIC), json.dumps(event_notification))
    return Gst.FlowReturn.OK

def train_classifier(events):
    logging.info('training classifier with events {}'.format(events))
    try:
        assert(len(events) > 1)
    except AssertionError:
        return(False, 'Number of events must be >= 2')

    # split recordings to 1-second files
    for event in events:
        recs_folder = os.path.join(ROOT_FOLDER, 'recordings', event)
        try:
            assert(os.path.exists(recs_folder))
        except AssertionError:
            logging.error('Recordings directory {} does not exist.',format(recs_folder))
            return (False, 'Could not find recordings directory {}'.format(recs_folder))

        out_folder = os.path.join(ROOT_FOLDER, 'samples', event)
        if os.path.exists(out_folder):
            logging.info('removing existing folder {}'.format(out_folder))
            shutil.rmtree(out_folder)

        logging.info('creating folder {}'.format(out_folder))
        os.makedirs(out_folder)

        wavs = [f for f in sorted(os.listdir(recs_folder)) if f.endswith('.wav')]
        offset = 0
        for wav in wavs:
            logging.info('splitting recording {}'.format(wav))
            offset = split_recording(os.path.join(recs_folder, wav), out_folder, offset)
            print('offset:', offset)

    # Train classifier from samples
    folders = [os.path.join(ROOT_FOLDER, 'samples', event) for event in events]
    ncls = len(populate_classifiers()) + 1
    classifier_name = 'racc_{:03d}'.format(ncls)
    logging.info('Creating classifier {}'.format(classifier_name))
    aT.featureAndTrain(folders, 1, 1, aT.shortTermWindow, aT.shortTermStep, 'svm', os.path.join(ROOT_FOLDER, 'classifiers', classifier_name), False)
    print('done training')
    return (True, 'test')


def start_classifier(classifier_name):
    global GST_PIPELINE, CLASSIFIER

    logging.info('starting classifier {}'.format(classifier_name))

    if GST_PIPELINE:
        GST_PIPELINE.set_state(Gst.State.NULL)

    logging.info('Loading event classifier')
    EVENT_CLASSIFIER['OBJ'], EVENT_CLASSIFIER['MEAN'], EVENT_CLASSIFIER['STD'], EVENT_CLASSIFIER['CLASSES'], _, _, _, _, _ =\
        aT.loadSVModel(os.path.join(ROOT_FOLDER, 'classifiers', classifier_name))
    logging.info('Succesfully loaded event classifier')
    GST_PIPELINE = classification_pipeline()
    sink = GST_PIPELINE.get_by_name('sink')
    sink.connect('new-sample', classification_callback)
    GST_PIPELINE.set_state(Gst.State.PLAYING)
    return (True, 'ok')

def stop_classifier():
    global GST_PIPELINE

    logging.info('stopping classifier')

    if GST_PIPELINE: # TODO: Check if this is a "Classification" pipeline
        GST_PIPELINE.set_state(Gst.State.NULL)
        GST_PIPELINE = None
    return (True, '')

# --------------------------------------------------
#              MQTT-related functions
# --------------------------------------------------
def init_mqtt_client(url, username=None, password=None):
    client = mqtt.Client()
    if username and password:
        client.username_pw_set(username, password)

    client.on_connect = lambda a,b,c,d: client.publish('{}/status'.format(ROOT_TOPIC), 'connected')
    client.message_callback_add('{}/cmd/ping'.format(ROOT_TOPIC), handle_ping_command)
    client.message_callback_add('{}/cmd/record'.format(ROOT_TOPIC), handle_record_command)
    client.message_callback_add('{}/cmd/classification'.format(ROOT_TOPIC), handle_classification_command)

    host, port = url.split(':')
    client.connect(host, int(port))

    client.subscribe('{}/cmd/#'.format(ROOT_TOPIC))
    return client

def respond(status, request_id, data=None):
    if status:
        payload = {'id': request_id, 'status': True, 'data': data}
    else:
        payload = {'id': request_id, 'status': False, 'msg': data}

    MQTT_CLIENT.publish('{}/response'.format(ROOT_TOPIC), json.dumps(payload), 1)

def handle_ping_command(client, userdata, message):
    try:
        msg = json.loads(message.payload.decode('utf-8'))
        req_id = msg['id'] if 'id' in msg else -1
        respond(True, req_id, 'pong')
    except:
        respond(True, -1, 'pong')

def handle_record_command(client, userdata, message):
    msg = json.loads(message.payload.decode('utf-8'))
    req_id = msg['id'] if 'id' in msg else -1
    logging.debug('Record request id: {}'.format(req_id))

    try:
        action = msg['action']
        logging.debug('Record action: {}'.format(action))
    except KeyError:
        logging.error('Malformed record command payload {}'.msg)
        respond(False, req_id, 'No action specified.')

    if action == 'start':
        try:
            logging.debug('Start recording')
            event = msg['event']
            logging.debug('Recording event: {}'.format(event))
            res = start_recording(event)
            respond(res[0], req_id, res[1])
        except Exception as e:
            respond(False, req_id, 'No event specified.')
    elif action == 'stop':
        logging.debug('Stop recording')
        res = stop_recording()
        respond(res[0], req_id, res[1])
    elif action == 'list':
        try:
            recordings = populate_recordings()
            respond(True, req_id, json.dumps(recordings))
        except:
            respond(False, req_id, 'Error while getting recordings')
    else:
        respond(False, req_id, 'Action must be one of [start, stop]')


def handle_classification_command(client, userdata, message):
    msg = json.loads(message.payload.decode('utf-8'))
    req_id = msg['id'] if 'id' in msg else -1
    action = msg['action'] if 'action' in msg else None
    classifier_name = msg['classifierName'] if 'classifierName' in msg else None
    
    print('MESSAgE:', msg)
    if not action:
        respond(False, req_id, 'No action specified.')


    if action == 'start':
        #try:
        logging.info('Starting classifier {}'.format(classifier_name))
        res = start_classifier(classifier_name)
        respond(res[0], req_id, res[1])
        #except Exception:
        #    respond(False, req_id, 'No classifier specified or classifier not found.')
    elif action == 'stop':
        res = stop_classifier()
        respond(res, req_id)
    elif action == 'create':
        logging.debug('create classifier command')
        try:
            events = msg['events']
            res = train_classifier(events)
            print('responding...', res[0], req_id, res[1])
            respond(res[0], req_id, res[1])
        except:
            print('in here!!')
            respond(False, req_id, 'No events specified or events recordings not found.')
    elif action == 'list':
        logging.debug('get classifiers list')
        try:
            classifiers = populate_classifiers()
            respond(True, req_id, json.dumps(classifiers))
        except:
            respond(False, req_id, 'Could not populate classifiers')
    else:
        respond(False, req_id, 'Unknown action {}'.format(action))


def main():
    global MQTT_CLIENT, RECORDINGS, AUDIO_DEVICE
    logging.getLogger().setLevel(logging.DEBUG)

    # make sure recordings + classifiers directories exist
    recordings_folder = os.path.join(ROOT_FOLDER, 'recordings')
    if not os.path.exists(recordings_folder):
        os.makedirs(recordings_folder)

    classifiers_folder = os.path.join(ROOT_FOLDER, 'classifiers')
    if not os.path.exists(classifiers_folder):
        os.makedirs(classifiers_folder)

    RECORDINGS = populate_recordings()
    logging.info('the recordings are: {}'.format(RECORDINGS))

    GObject.threads_init()
    Gst.init(None)
    version_str = '.'.join(map(str, Gst.version()))
    print('Using GStreamer version {}'.format(version_str))

    args = parse_arguments()
    AUDIO_DEVICE = args.device
    MQTT_CLIENT = init_mqtt_client(args.url, args.username, args.password)
    #MQTT_CLIENT.loop_start()
    MQTT_CLIENT.loop_forever()

if __name__ == '__main__':
    main()

    #while raw_input('\033[1;33;40mPress [q] to exit\033[0m ') != 'q':
    #    pass

    print('Bye.')
    #if MQTT_CLIENT:
    #    MQTT_CLIENT.loop_stop()
    #    MQTT_CLIENT.disconnect()
    if GST_PIPELINE:
        GST_PIPELINE.set_state(Gst.State.NULL)
    sys.exit(0)
