# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from django.shortcuts import render
from django.http import HttpResponse
import json

import argparse
import os.path
import re
import sys
import tarfile
import time

# Remove warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
from six.moves import urllib

import cv2
import datetime
from threading import Thread

image_file = ''
model_dir = 'imagedataset'
num_top_predictions = 1
imageEvalStack = []
imageEvalStackCtr = 0
imgCapIsRunning = True
completeEvalImageCtr = 0
isLoading = False;
latestImage = ''
fileLocation = 'External/cameraimgs/'
isCapturing = True

latestEvalImageClass = ''
latestEvalImageScore = 0

liveStreamDeleteDelay = 20
liveStreamKeep = 30
latestImagex = ''

def imageCapture():
	global imageEvalStackCtr
	global imageEvalStack
	global imgCapIsRunning
	global isLoading
	global latestImage
	global fileLocation
	global isCapturing
	global latestImagex
	liveStreamOn = True
	camCapture = cv2.VideoCapture(1)
	camCapture.set(cv2.CAP_PROP_FPS, 30)
	prevTime = datetime.datetime.now()
	nCtr = 0;
	timestampnow = ''
	if camCapture.isOpened():
		while isCapturing:
		# while nCtr < 2:
			# try:
			ret, imgCapture = camCapture.read()
			curTime = datetime.datetime.now()
			
			# if curTime.second != prevTime.second:
			prevTime = datetime.datetime.now()
			filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
			cv2.imwrite(fileLocation + filename, imgCapture)
			completefilename = 'external/cameraimgs/' + filename
			latestImage = completefilename
			
			if not isLoading :
				isLoading = True;
				resizedImage = cv2.resize(imgCapture, (160, 120))

				cv2.imwrite(fileLocation + 'x' + filename, resizedImage)
				latestImagex = 'external/cameraimgs/' + 'x' + filename
				nCtr += 1
				# runEval = Thread(target=evaluate(fileLocation + 'x' + filename))
				# runEval.start()
				imageEvalStack.append(fileLocation + 'x' + filename)
				imageEvalStackCtr += 1

			# except:
				# pass
				# print("Error Saving Image")

			key = cv2.waitKey(50)
			if key == 27:
				del(camCapture)
				break
		del(camCapture)
		imgCapIsRunning = False

def imageEvaluator():
	global imageEvalStack
	global imageEvalStackCtr
	global completeEvalImageCtr

	currentImageEvalStackCtr = 0
	while imgCapIsRunning or completeEvalImageCtr != imageEvalStackCtr:
		if(currentImageEvalStackCtr < imageEvalStackCtr):
			evaluate(imageEvalStack[currentImageEvalStackCtr])
			currentImageEvalStackCtr += 1

def evaluate(image_file):
	global completeEvalImageCtr
	global isLoading
	image = (image_file if image_file else os.path.join(model_dir, image_file))
	completeEvalImageCtr += 1
	isLoading = False
# camera.deleteThread.start()
#=======================================================================
# FLAGS = None

class NodeLookup(object):
	"""Converts integer node ID's to human readable labels."""

	def __init__(self,
							 label_lookup_path=None,
							 uid_lookup_path=None):
		if not label_lookup_path:
			label_lookup_path = os.path.join(
					model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
		if not uid_lookup_path:
			uid_lookup_path = os.path.join(
					model_dir, 'imagenet_synset_to_human_label_map.txt')
		self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

	def load(self, label_lookup_path, uid_lookup_path):
		"""Loads a human readable English name for each softmax node.

		Args:
			label_lookup_path: string UID to integer node ID.
			uid_lookup_path: string UID to human-readable string.

		Returns:
			dict from integer node ID to human-readable string.
		"""
		if not tf.gfile.Exists(uid_lookup_path):
			tf.logging.fatal('File does not exist %s', uid_lookup_path)
		if not tf.gfile.Exists(label_lookup_path):
			tf.logging.fatal('File does not exist %s', label_lookup_path)

		# Loads mapping from string UID to human-readable string
		proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
		uid_to_human = {}
		p = re.compile(r'[n\d]*[ \S,]*')
		for line in proto_as_ascii_lines:
			parsed_items = p.findall(line)
			uid = parsed_items[0]
			human_string = parsed_items[2]
			uid_to_human[uid] = human_string

		# Loads mapping from string UID to integer node ID.
		node_id_to_uid = {}
		proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
		for line in proto_as_ascii:
			if line.startswith('  target_class:'):
				target_class = int(line.split(': ')[1])
			if line.startswith('  target_class_string:'):
				target_class_string = line.split(': ')[1]
				node_id_to_uid[target_class] = target_class_string[1:-2]

		# Loads the final mapping of integer node ID to human-readable string
		node_id_to_name = {}
		for key, val in node_id_to_uid.items():
			if val not in uid_to_human:
				tf.logging.fatal('Failed to locate: %s', val)
			name = uid_to_human[val]
			node_id_to_name[key] = name

		return node_id_to_name

	def id_to_string(self, node_id):
		if node_id not in self.node_lookup:
			return ''
		return self.node_lookup[node_id]

runonce = 0
def liveDelete():
	global fileLocation
	while isCapturing:
		dir_to_search = fileLocation
		for dirpath, dirnames, filenames in os.walk(dir_to_search):
			filenames.sort(reverse=True)
			for file in filenames[liveStreamKeep:]:
				curpath = os.path.join(dirpath, file)
				os.remove(curpath)
		time.sleep(liveStreamDeleteDelay)

def toggleCapture(request):
	global isCapturing
	JSONer = {}
	JSONer['output'] = "Doing Nothing..."
	
	if isCapturing:
		isCapturing = False
		JSONer['output'] = "Turn Capture ON"
	else:
		isCapturing = True
		JSONer['output'] = "Turn Capture OFF"

	return HttpResponse(json.dumps(JSONer))

def getLatestImage(request):
	global latestImage
	global latestImagex
	JSONer = {}
	JSONer['cam_view'] = 'http://10.237.228.79:80/' + latestImage
	JSONer['eval_view'] = 'http://10.237.228.79:80/' + latestImagex
	JSONer['output'] = "OUTPUT LALALA"
	JSONer['output_count'] = completeEvalImageCtr
	# print(latestEvalImageScore)
	JSONer['latestEvalImageClass'] = latestEvalImageClass
	JSONer['latestEvalImageScore'] = int(latestEvalImageScore * 100)
	print(int(latestEvalImageScore * 100))
	return HttpResponse(json.dumps(JSONer))

streamThread = Thread(target=imageCapture)
streamThread.start()
deleteThread = Thread(target=liveDelete)
deleteThread.start()
evalThread = Thread(target=imageEvaluator)
evalThread.start()

def defaultPage(request):
	print("DEFAULT PAGE")
	return render(request, 'home.html', {'variable':'Hello World'})



