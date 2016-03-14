from guineapig import *
import sys
import re
import math

# supporting routines can go here
def tokens(line):
	elements = line.split('\t')
	for label in elements[1].split(','):
#		yield label
		for tok in elements[2].split():
			tok = re.sub('\W','',tok)
			if len(tok)>0:
				yield (tok,label)

def labels(line):
	elements = line.split('\t')
	for label in elements[1].split(','):
		yield label

def formatReq(line):
	elements = line.split('\t')
	for tok in elements[2].split():
		tok = re.sub('\W', '', tok)
		if len(tok)>0:
			yield (tok,elements[0])

		
#always subclass Planner
class NB(Planner):
	# params is a dictionary of params given on the command line. 
	# e.g. trainFile = params['trainFile']
	params = GPig.getArgvParams()
	trainLines= ReadLines(params['trainFile'])

	tokenCounts = Flatten(trainLines, by=tokens) | Group(by=lambda x:x, reducingTo=ReduceToCount())

	labelCounts = Flatten(trainLines, by=labels) | Group(by=lambda x:x, reducingTo=ReduceToCount())
        
	vocSize = Map(tokenCounts, by=lambda ((t,l), c):t) | Distinct() | Group(by=lambda row:'vocSize', reducingTo=ReduceToCount())
	
	labelTotTokens = Group(tokenCounts, by=lambda ((t,l), c):l, retaining=lambda((t,l), c):c, reducingTo=ReduceToSum())

	totLabels = Group(labelCounts, by=lambda (l, c):'totLabels', retaining=lambda(l,c):c, reducingTo=ReduceToSum())

	stats = Augment(Join(Jin(labelTotTokens, by=lambda(l,tCount):l), Jin(labelCounts, by=lambda(label,lCount):label)), sideview=vocSize, loadedBy=lambda v:GPig.onlyRowOf(v)) | Augment(sideview=totLabels, loadedBy=lambda v:GPig.onlyRowOf(v)) | ReplaceEach(by=lambda((((l,tCount),(label,lCount)),(alsodummy,vocSize)),(dummy,totL)):(l,tCount,lCount,vocSize,totL)) 

	#tokenMap = Map(tokenCounts, by=lambda (w, lc):(w.split('|')[0], w.split('|')[1], lc))

	#tokenMapEx = Augment(Join(Jin(tokenCounts, by=lambda((w,l),c):l), Jin(labelTotTokens, by=lambda(label,count):label)), sideview=vocSize, loadedBy=lambda v:GPig.onlyRowOf(v)) | ReplaceEach(by=lambda((((w,l),c),(label,count)),(dummy,n)):(l, w, c, count, n))
	
	invertedList = Group(tokenCounts, by=lambda((t,l),c):t, retaining=lambda((t,l),c):(l, math.log(float(c+1))))

	req = ReadLines(params['testFile']) | Flatten(by=formatReq)
	testDocWordCount = Group(req, by=lambda(w,docid):docid, reducingTo=ReduceToCount())

	join = Join(Jin(invertedList, by=lambda(t, lc):t), Jin(req, by=lambda (w, docid):w)) | ReplaceEach(by=lambda((t, lc), (w, docid)):(t, docid, lc))

	docidLabelProb = FlatMap(join, by=lambda (w,docid,lp):map(lambda (l,p):((docid, l), p), lp)) | Group(by=lambda ((docid, l), p):(docid, l), retaining=lambda((docid, l),p):p, reducingTo=ReduceToSum())
	
	docidLabelProbEx = Join(Jin(docidLabelProb, by=lambda((docid, l),p):docid), Jin(testDocWordCount, by=lambda(d,c):d))

	joinLabel = Join(Jin(docidLabelProbEx, by=lambda(((docid,l),p),(d,c)):l), Jin(stats, by=lambda(label,tCount, lCount,vocSize,totL):label)) | ReplaceEach(by=lambda((((docid,l),p),(d,c)),(label,tCount, lCount,vocSize,totL)):(docid, l, p-c*math.log(float(tCount+vocSize))+math.log(float(lCount)+0.05*vocSize)-math.log(float(totL+vocSize))))

	output = Group(joinLabel, by=lambda (docid, l, p):docid, retaining=lambda(docid, l, p):(p,l)) | ReplaceEach(by=lambda (docid, llist):(docid,max(llist))) | ReplaceEach(by=lambda (docid, (p, l)):(docid, l, p))
	#Augment(join, sideview=totLabels, loadedBy=lambda v:GPig.onlyRowOf(v))

# always end like this
if __name__ == "__main__":
    NB().main(sys.argv)

# supporting routines can go here

