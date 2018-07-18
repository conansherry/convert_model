#!/usr/bin/python
import hashlib
import os
import sys

if len(sys.argv) < 2:
	sys.exit('Usage: %s filename' % sys.argv[0])


if not os.path.exists(sys.argv[1]):
	sys.exit('ERROR: File "%s" was not found!' % sys.argv[1])

with open(sys.argv[1], 'rb') as f:
	contents = f.read()
	print("SHA1: %s" % hashlib.sha1(contents).hexdigest())
	print("SHA256: %s" % hashlib.sha256(contents).hexdigest())
	#md5 accepts only chunks of 128*N bytes
	md5 = hashlib.md5()
	for i in range(0, len(contents), 8192):
		md5.update(contents[i:i+8192])
	print("MD5: %s" % md5.hexdigest())