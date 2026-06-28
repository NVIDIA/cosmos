# flake8: noqa
#!/usr/bin/env python3

#File: rrc_evaluation_funcs_1_1.py
#Version: 1.1
#Version info: changes for Python 3
#Date: 2019-12-29
#Description: File with useful functions to use by the evaluation scripts in the RRC website.

import json
import sys

sys.path.append('./')
import importlib
import os
import re
import zipfile


def print_help():
    sys.stdout.write('Usage: python %s.py -g=<gtFile> -s=<submFile> [-o=<outputFolder> -p=<jsonParams>]' %sys.argv[0])
    sys.exit(2)


def load_zip_file_keys(file,fileNameRegExp=''):
    """
    Returns an array with the entries of the ZIP file that match with the regular expression.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    """
    try:
        archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except :
        raise Exception('Error loading the ZIP archive.')

    pairs = []

    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)

        if addFile:
            pairs.append( keyName )

    return pairs


def load_zip_file(file,fileNameRegExp='',allEntries=False):
    """
    Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    allEntries validates that all entries in the ZIP file pass the fileNameRegExp
    """
    try:
        archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except :
        raise Exception('Error loading the ZIP archive')

    pairs = []
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)

        if addFile:
            pairs.append( [ keyName , archive.read(name)] )
        else:
            if allEntries:
                raise Exception('ZIP entry not valid: %s' %name)

    return dict(pairs)

def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        return raw.decode('utf-8-sig',errors = 'replace')
    except:
       return None

def validate_lines_in_file(fileName,file_contents,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    This function validates that all lines of the file calling the Line validation function for each line
    """
    utf8File = decode_utf8(file_contents)
    if (utf8File is None) :
        raise Exception("The file %s is not UTF-8" %fileName)

    lines = utf8File.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != ""):
            try:
                validate_tl_line(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)
            except Exception as e:
                raise Exception(("Line in sample not valid. Sample: %s Line: %s Error: %s" %(fileName,line,str(e))).encode('utf-8', 'replace'))



def validate_tl_line(line,LTRB=True,withTranscription=True,withConfidence=True,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    """
    get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)


def get_tl_line_values(line,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    Returns values from a textline. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = "";
    points = []

    numPoints = 4;

    if LTRB:

        numPoints = 4;

        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence,transcription")
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence")
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax,transcription")
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax")

        xmin = int(m.group(1))
        ymin = int(m.group(2))
        xmax = int(m.group(3))
        ymax = int(m.group(4))
        if(xmax<xmin):
                raise Exception("Xmax value (%s) not valid (Xmax < Xmin)." %(xmax))
        if(ymax<ymin):
                raise Exception("Ymax value (%s)  not valid (Ymax < Ymin)." %(ymax))

        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]

        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(xmin,ymin,imWidth,imHeight);
            validate_point_inside_bounds(xmax,ymax,imWidth,imHeight);

    else:

        numPoints = 8;

        if withTranscription and withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription")
        elif withConfidence:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence")
        elif withTranscription:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription")
        else:
            m = re.match(r'^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$',line)
            if m == None :
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4")

        points = [ float(m.group(i)) for i in range(1, (numPoints+1) ) ]

        validate_clockwise_points(points)

        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(points[0],points[1],imWidth,imHeight);
            validate_point_inside_bounds(points[2],points[3],imWidth,imHeight);
            validate_point_inside_bounds(points[4],points[5],imWidth,imHeight);
            validate_point_inside_bounds(points[6],points[7],imWidth,imHeight);


    if withConfidence:
        try:
            confidence = float(m.group(numPoints+1))
        except ValueError:
            raise Exception("Confidence value must be a float")

    if withTranscription:
        posTranscription = numPoints + (2 if withConfidence else 1)
        transcription = m.group(posTranscription)
        m2 = re.match(r'^\s*\"(.*)\"\s*$',transcription)
        if m2 != None : #Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")

    return points,confidence,transcription

def get_tl_dict_values(detection,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,validNumPoints=[],validate_cw=True):
    """
    Validate the format of the dictionary. If the dictionary is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values:
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]]}
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4,"illegibility":false}
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4,"dontCare":false}
    Returns values from the dictionary. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = "";
    points = []

    if isinstance(detection, dict) == False :
        raise Exception("Incorrect format. Object has to be a dictionary")

    if not 'points' in detection:
        raise Exception("Incorrect format. Object has no points key)")

    if isinstance(detection['points'], list) == False :
        raise Exception("Incorrect format. Object points key have to be an array)")

    num_points = len(detection['points'])

    if num_points<3 :
        raise Exception("Incorrect format. Incorrect number of points. At least 3 points are necessary. Found: " +  str(num_points))

    if(len(validNumPoints)>0 and num_points in validNumPoints == False ):
        raise Exception("Incorrect format. Incorrect number of points. Only allowed 4,8 or 12 points)")

    for i in range(num_points):
        if isinstance(detection['points'][i], list) == False :
            raise Exception("Incorrect format. Point #" + str(i+1) + " has to be an array)")

        if len(detection['points'][i]) != 2 :
            raise Exception("Incorrect format. Point #" + str(i+1) + " has to be an array with 2 objects(x,y) )")

        if isinstance(detection['points'][i][0], (int,float) ) == False or isinstance(detection['points'][i][1], (int,float) ) == False :
            raise Exception("Incorrect format. Point #" + str(i+1) + " childs have to be Integers)")

        if (imWidth>0 and imHeight>0):
            validate_point_inside_bounds(detection['points'][i][0],detection['points'][i][1],imWidth,imHeight);

        points.append(float(detection['points'][i][0]))
        points.append(float(detection['points'][i][1]))

    if validate_cw :
        validate_clockwise_points(points)

    if withConfidence:
        if not 'confidence' in detection:
            raise Exception("Incorrect format. No confidence key)")

        if isinstance(detection['confidence'], (int,float)) == False :
            raise Exception("Incorrect format. Confidence key has to be a float)")

        if detection['confidence']<0 or detection['confidence']>1 :
            raise Exception("Incorrect format. Confidence key has to be a float between 0.0 and 1.0")

        confidence = detection['confidence']

    if withTranscription:
        if not 'transcription' in detection:
            raise Exception("Incorrect format. No transcription key)")

        if isinstance(detection['transcription'], str) == False :
            raise Exception("Incorrect format. Transcription has to be a string. Detected: " + type(detection['transcription']).__name__ )

        transcription = detection['transcription']

        if 'illegibility' in detection: #Ensures that if illegibility atribute is present and is True the transcription is set to ### (don't care)
            if detection['illegibility'] == True:
                transcription = "###"

        if 'dontCare' in detection: #Ensures that if dontCare atribute is present and is True the transcription is set to ### (don't care)
            if detection['dontCare'] == True:
                transcription = "###"

    return points,confidence,transcription

def validate_point_inside_bounds(x,y,imWidth,imHeight):
    if(x<0 or x>imWidth):
            raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" %(xmin,imWidth,imHeight))
    if(y<0 or y>imHeight):
            raise Exception("Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s" %(ymin,imWidth,imHeight))

def validate_clockwise_points(points):
    """
    Validates that the points are in clockwise order.
    """
    edge = []
    for i in range(len(points)//2):
        edge.append( (int(points[(i+1)*2 % len(points)]) - int(points[i*2])) * (int(points[ ((i+1)*2+1) % len(points)]) + int(points[i*2+1]))   )
    if sum(edge)>0:
        raise Exception("Points are not clockwise. The coordinates of bounding points have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")

def get_tl_line_values_from_file_contents(content,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True):
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    lines = content.split( "\r\n" if CRLF else "\n" )
    for line in lines:
        line = line.replace("\r","").replace("\n","")
        if(line != "") :
            points, confidence, transcription = get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight);
            pointsList.append(points)
            transcriptionsList.append(transcription)
            confidencesList.append(confidence)

    if withConfidence and len(confidencesList)>0 and sort_by_confidences:
        import numpy as np
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    return pointsList,confidencesList,transcriptionsList

def get_tl_dict_values_from_array(array,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True,validNumPoints=[],validate_cw=True):
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid dict formats:
    {"points":[[x1,y1],[x2,y2],[x3,x3],..,[xn,yn]],"transcription":"###","confidence":0.4}
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    for n in range(len(array)):
        objectDict = array[n]
        points, confidence, transcription = get_tl_dict_values(objectDict,withTranscription,withConfidence,imWidth,imHeight,validNumPoints,validate_cw);
        pointsList.append(points)
        transcriptionsList.append(transcription)
        confidencesList.append(confidence)

    if withConfidence and len(confidencesList)>0 and sort_by_confidences:
        import numpy as np
        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    return pointsList,confidencesList,transcriptionsList

def main_evaluation(p,default_evaluation_params_fn,validate_data_fn,evaluate_method_fn,show_result=True,per_sample=True):
    """
    This process validates a method, evaluates it and if it succed generates a ZIP file with a JSON entry for each sample.
    Params:
    p: Dictionary of parmeters with the GT/submission locations. If None is passed, the parameters send by the system are used.
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    evaluate_method_fn: points to a function that evaluated the submission and return a Dictionary with the results
    """

    if (p == None):
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        if(len(sys.argv)<3):
            print_help()

    evalParams = default_evaluation_params_fn()
    if 'p' in p.keys():
        evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p']) )

    resDict={'calculated':True,'Message':'','method':'{}','per_sample':'{}'}
    try:
        validate_data_fn(p['g'], p['s'], evalParams)
        evalData = evaluate_method_fn(p['g'], p['s'], evalParams)
        resDict.update(evalData)

    except Exception as e:
        resDict['Message']= str(e)
        resDict['calculated']=False

    if 'o' in p:
        if not os.path.exists(p['o']):
            os.makedirs(p['o'])

        resultsOutputname = p['o'] + '/results.zip'
        outZip = zipfile.ZipFile(resultsOutputname, mode='w', allowZip64=True)

        del resDict['per_sample']
        if 'output_items' in resDict.keys():
            del resDict['output_items']

        outZip.writestr('method.json',json.dumps(resDict))

    if not resDict['calculated']:
        if show_result:
            sys.stderr.write('Error!\n'+ resDict['Message']+'\n\n')
        if 'o' in p:
            outZip.close()
        return resDict

    if 'o' in p:
        if per_sample == True:
            for k,v in evalData['per_sample'].items():
                outZip.writestr( k + '.json',json.dumps(v))

        if 'output_items' in evalData.keys():
            for k, v in evalData['output_items'].items():
                outZip.writestr( k,v)

        outZip.close()

    # if show_result:
    #     #sys.stdout.write("Calculated!")
    #     sys.stdout.write(json.dumps(resDict['method']))

    return resDict


def main_validation(default_evaluation_params_fn,validate_data_fn):
    """
    This process validates a method
    Params:
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    """
    try:
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        evalParams = default_evaluation_params_fn()
        if 'p' in p.keys():
            evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p']) )

        validate_data_fn(p['g'], p['s'], evalParams)
        print ('SUCCESS')
        sys.exit(0)
    except Exception as e:
        print (str(e))
        sys.exit(101)
