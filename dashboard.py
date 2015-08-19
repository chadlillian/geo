#!/home/chad/anaconda/bin/python

import	sys
import	urllib2
import	urllib
import	requests
import	time
import	numpy as np

class	dashboard:
	def	__init__(self):
		self.url	='http://blinkinglightsmonitor.appspot.com'
	
	def	setURL(self,url):
		self.url	=url.rstrip('/')

	def	update(self,data):
		args	='/update?'+'&'.join(["%s=%s"%(k,data[k]) for k in data.keys()])
		requests.get(self.url+args)

if __name__=="__main__":
	dash	=dashboard()
	dash.setURL('http://localhost:8080')

	for i in range(10):
		n	=np.random.randint(100)
		
		dash.update({'value':n,'application':'X'})
		time.sleep(2)
