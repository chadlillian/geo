import  cgi
import  webapp2
import  jinja2
import  os
import  datetime
import  json
#from google.appengine.ext import ndb
#from google.appengine.api import users

#jinja_env  =jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))


class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.out.write(open('poly.html').read())

class   Data(webapp2.RequestHandler):
    def get(self):
        dd   =open('sample-geojson-poly.js').read()
        #dd   =open('sample-geojson-cluster2.js').read()
        self.response.out.write(dd)

app = webapp2.WSGIApplication(  [('/', MainPage),
                                ('/geojson.js',Data),
                                ],
                                debug=True)

