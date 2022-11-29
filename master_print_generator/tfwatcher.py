""" A module to maintain a watching list of tensorflow tensors
The watch list is maintained as a global variable that can be updated whenever
this module is imported. This could save a lot of time compared with passing all
the tensorfow tensors as function arguments.
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

watchlist = {}
conflict_solution = "latest" # solution when key conflicts, one of followings ["oldest", "latest", "exception"]

def insert(key, var):
	assert type(key) is str, "the key for tfwatcher must be an string!"
	if key in watchlist.keys():
		if conflict_solution == "oldest":
			pass
		elif conflict_solution == "latest":
			watchlist[key] = var
		elif conflict_solution == "exception":
			raise KeyError("Trying to insert an variable with key {} \
				when it is already in the watchlist of tfwatcher!".format(key))
		else:
			raise ValueError("Unknown conflict_solution in tfwatcher: {}".format(conflict_solution))
	else:
		watchlist[key] = var

def get_watchlist():
	return watchlist
