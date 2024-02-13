#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil

from modules.user import User


if __name__ == '__main__':
	splits = ('train', 'valid', 'test')
	parser = argparse.ArgumentParser("./infer.py")
	parser.add_argument(
		'--dataset', '-d',
		type=str,
		required=True,
		help='Dataset to train with. No Default',
	)
	parser.add_argument(
		'--config',
		type=str,
		required=False,
		default='config/RangeRet-semantickitti.yaml',
		help='Architecture yaml cfg file. See /config/ for sample. No default!',
	)
	parser.add_argument(
		'--data',
		type=str,
		required=False,
		default='config/labels/semantic-kitti.yaml',
		help='Classification yaml cfg file. See /config/labels for sample. No default!',
	)
	parser.add_argument(
		'--log', '-l',
		type=str,
		default=os.getcwd() + '/log/' +
		datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
		help='Directory to put the predictions. Default: log/date+time'
	)
	parser.add_argument(
		'--model', '-m',
		type=str,
		required=True,
		default=None,
		help='Directory to get the trained model.'
	)
	parser.add_argument(
        '--split', '-s',
        type=str,
        required=True,
        default='valid',
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
	FLAGS, unparsed = parser.parse_known_args()

	# print summary of what we will do
	print("----------")
	print("INTERFACE:")
	print("dataset", FLAGS.dataset)
	print("config", FLAGS.config)
	print("data", FLAGS.data)
	print("log", FLAGS.log)
	print("model", FLAGS.model)
	print("split", FLAGS.split)
	print("----------\n")
	print("Commit hash (training version): ", str(
		subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
	print("----------\n")

	# open arch config file
	try:
		print("Opening arch config file from %s" % FLAGS.config)
		ARCH = yaml.safe_load(open(FLAGS.config,'r'))
	except Exception as e:
		print(e)
		print("Error opening arch yaml file.")
		quit()

	# open data config file
	try:
		print("Opening data config file from %s" % FLAGS.data)
		DATA = yaml.safe_load(open(FLAGS.data, 'r'))
	except Exception as e:
		print(e)
		print("Error opening data yaml file.")
		quit()

	# create log folder
	try:
		if os.path.isdir(FLAGS.log):
			shutil.rmtree(FLAGS.log)
		os.makedirs(FLAGS.log)
		os.makedirs(os.path.join(FLAGS.log, "sequences"))
		for seq in DATA["split"]["train"]:
			seq = '{0:02d}'.format(int(seq))
			print("train", seq)
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
		for seq in DATA["split"]["valid"]:
			seq = '{0:02d}'.format(int(seq))
			print("valid", seq)
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
		for seq in DATA["split"]["test"]:
			seq = '{0:02d}'.format(int(seq))
			print("test", seq)
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
			os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
	except Exception as e:
		print(e)
		print("Error creating log directory. Check permissions!")
		raise

	# does model folder exist?
	if os.path.isdir(FLAGS.model):
		print("model folder exists! Using model from %s" % (FLAGS.model))
	else:
		print("model folder doesnt exist! Can't infer...")
		quit()

	# create user and infer dataset
	user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, FLAGS.split)
	user.infer()