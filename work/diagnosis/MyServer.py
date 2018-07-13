#!/usr/bin/python
# coding=utf-8
import os,sys,json,TextSearch,EmrDiag
from werkzeug.wrappers import Request, Response
from werkzeug.routing import Map, Rule
from werkzeug.exceptions import HTTPException, NotFound
from werkzeug.wsgi import SharedDataMiddleware
import logging
import logging.config
import ConfigParser
root_path=sys.path[0] + os.sep
class Shortly(object):
    ts = TextSearch.TextSearch()
    emrdiag = EmrDiag.Emrdiag()
    def __init__(self, config):
        self.config = ConfigParser.ConfigParser()
        self.config.read(root_path + 'Config.properties')
        logging.config.fileConfig(root_path + 'logging.conf')
        root_logger = logging.getLogger('root')
        root_logger.debug('test root logger...')

        logger = logging.getLogger('main')
        logger.info("start init...")
        self.url_map = Map([
            Rule('/', endpoint='search'),
            Rule('/search', endpoint='search'),
            Rule('/search1', endpoint='search1'),
            Rule('/result', endpoint='emrdiag'),
            Rule('/diagnosis', endpoint='diagnosis'),
        ])
        #print self.ts
        print("end init...")

    def dispatch_request(self, request):
        adapter = self.url_map.bind_to_environ(request.environ)
        try:
            endpoint, values = adapter.match()
            return getattr(self, 'on_' + endpoint)(request, **values)
        except NotFound, e:
            #return self.error_404()
            return e
        except HTTPException, e:
            return e

    def wsgi_app(self, environ, start_response):
        request = Request(environ)
        response = self.dispatch_request(request)
        return response(environ, start_response)

    def __call__(self, environ, start_response):
        return self.wsgi_app(environ, start_response)


    def on_search(self,request):
        subject = str(getParameter(request,"subject",""))
        condition = str(getParameter(request,"condition","")).decode('utf8')
        print subject,condition
        result = self.ts.search(condition,subject)
        response = []
        icd_list_temp = ['J15','J20','J45','T17','A16','A17','J06']
        temp_list = []

        for m in icd_list_temp:
            for row in result['data']:
                r_icd = row['ICD'].replace('\'','')
                if m == r_icd:
                    temp_list.append(row['ICD'])
                    response.append({'id': row['DocID'], 'name': row['DocName'], 'ICD': row['ICD'], 'rank': row['R']})
                if len(temp_list)>20:
                    temp_list = []
                    break
        return Response(json.dumps({ 'TotalCount': len(result['data']), 'Data': response, 'Time': result['milliseconds'] }), mimetype='application/json')


    def on_search1(self,request):
        subject = str(getParameter(request,"subject",""))
        condition = str(getParameter(request,"condition","")).decode('utf8')
        print subject,condition
        result = self.ts.search(condition,subject)
        #print result
        response = []
        icd_list = []
        for row in result['data']:
            if row['ICD'] not in icd_list:
                icd_list.append(row['ICD'])
                response.append({'id': row['DocID'], 'name': row['DocName'], 'ICD': row['ICD'], 'rank': row['R']})
        return Response(json.dumps({ 'TotalCount': len(result['data']), 'Data': response, 'Time': result['milliseconds'] }), mimetype='application/json')

    def on_diagnosis(self,request):
        subject = str(getParameter(request,"subject",""))
        condition = str(getParameter(request,"condition","")).decode('utf8')
        method = str(getParameter(request,"method","")).decode('utf8')
        is_return_emrs = str(getParameter(request,"is_return_emrs","")).decode('utf8')
        if method == 'None':
            method = 'distance'
        if is_return_emrs == 'None':
            is_return_emrs = False
        print subject, condition, method, is_return_emrs
        result = self.ts.diagnosis(condition, subject, method=method, is_penalize=True, is_return_emrs=is_return_emrs)
        #print result
        return Response(json.dumps(result), mimetype='application/json')

    def on_emrdiag(self,request):
        messagemap = {}
        messagemap['status'] = 200
        messagemap['code'] = 0
        messagemap['message'] = ''
        status = str(getParameter(request,"status",""))
        inCOD = str(getParameter(request,"inCOD",""))
        mainCheck = str(getParameter(request,"mainCheck",""))
        geneticHistory = str(getParameter(request,"geneticHistory",""))
        bloodTest = str(getParameter(request,"bloodTest",""))
        bloodTestVersion = str(getParameter(request,"bloodTestVersion",""))
        data = status + inCOD + mainCheck + geneticHistory + bloodTest + bloodTestVersion
        messagemap, output = self.emrdiag.hospital(data)
        return Response('json_success({' + '"data": ' + json.dumps(output) +  ',"message":' + json.dumps(messagemap) + '});', mimetype='application/json')
def getParameter(request,inputName,defaultValue):
        value = request.args.get(inputName)
        if value is None:
            value = request.form.get(inputName)
        return value


def create_app(redis_host='localhost', redis_port=6379, with_static=True):
    app = Shortly({
        'redis_host':       redis_host,
        'redis_port':       redis_port
    })
    if with_static:
        app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
            '/static':  os.path.join(os.path.dirname(__file__), 'static')
        })
    return app

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    reload(sys)
    sys.setdefaultencoding("utf-8")
    app = create_app()
    run_simple('0.0.0.0', 8088, app, use_debugger=True, use_reloader=False)
