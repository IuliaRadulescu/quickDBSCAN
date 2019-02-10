# coding: utf-8

__author__ = "Radulescu Iulia-Maria"
__copyright__ = "Copyright 2017, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "iulia.radulescu@cs.pub.ro"
__status__ = "Production"

import numpy as np
import cv2

import sys
import collections
from random import randint
import pymongo

import mongoConnect

class quickDBSCAN:
	
	def __init__(self, filePath):
		img = cv2.imread(filePath, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		imgReshaped = np.reshape(img, (len(img)*len(img[0]), 3))
		self.pixelList = list(np.unique(imgReshaped, axis=0))
		self.mongoConnectInstance = quickDBSCANMongo("QuickDBScanDB");
		#print(self.pixelList)

	
	def euclideanDist(self, pixel1, pixel2):
		#print(type(pixel1))
		#print(type(pixel2))
		return np.linalg.norm(pixel1-pixel2)
		
	def nestedLoop(self, eps, objs):
		for pixel1 in objs:
			for pixel2 in objs:
				if( (pixel1 == pixel2).all() == False and self.euclideanDist(pixel1, pixel2) <= eps):
					print(pixel1, pixel2)
					#insert into Mongo
					self.mongoConnectInstance.upsertPixelValue("quickDBSCAN",{"object":pixel1.tolist()}, pixel2.tolist())

	def nestedLoop2(self, eps, objs1, objs2):
		for pixel1 in objs1:
			for pixel2 in objs2:
				if( (pixel1 == pixel2).all() == False and self.euclideanDist(pixel1, pixel2) <= eps):
					print(pixel1, pixel2)
					#insert into Mongo
					self.mongoConnectInstance.upsertPixelValue("quickDBSCAN",{"object":pixel1.tolist()}, pixel2.tolist())


	def randomObject(self, objs, pixel):
		if(type(pixel) is not int):
			#print(objs)
			objs = [x for x in objs if (x == pixel).all() == False]
		
		randomIndex = randint(0, len(objs)-1)

		return objs[randomIndex]

					
	def partition(self, eps, objs, p1, p2):
		partL = [] 
		partG = []
		winL = []
		winG = []
		r = self.euclideanDist(p1, p2)
		startIdx = 0
		endIdx = len(objs)-1
		startDist = self.euclideanDist(objs[startIdx], p1)
		endDist = self.euclideanDist(objs[endIdx], p1)
		
		while(startIdx < endIdx):
			while(endDist > r and startIdx < endIdx):
				if(endDist <= r+eps):
					winG.append(objs[endIdx])
				endIdx = endIdx - 1
				endDist = self.euclideanDist(objs[endIdx], p1)
				
			while(startDist <= r and startIdx < endIdx):
				if(startDist >= r-eps):
					winL.append(objs[startIdx])
				startIdx = startIdx + 1
				startDist = self.euclideanDist(objs[startIdx], p1)
				
			if(startIdx < endIdx):
				if(endDist >= r-eps):
					winL.append(objs[endIdx])
				if(startDist <= r+eps):
					winG.append(objs[startIdx])
				#exchange items
				objs[startIdx], objs[endIdx] = objs[endIdx], objs[startIdx]
				startIdx = startIdx + 1
				endIdx = endIdx - 1
				startDist = self.euclideanDist(objs[startIdx], p1)
				endDist = self.euclideanDist(objs[endIdx], p1)
		
		if(startIdx == endIdx):
			if(endDist > r and endDist <= r+eps):
				winG.append(objs[endIdx])
			if(startDist <= r and startDist >= r-eps):
				winL.append(objs[startIdx])
			if(endDist > r):
				endIdx = endIdx - 1
				
		return (objs[0:endIdx], objs[endIdx+1:len(objs)-1], winL, winG)
	
				
		
	def quickJoin(self, objs, eps, constSmallNumber):
		if(len(objs) < constSmallNumber):
			self.nestedLoop(eps, objs)
			return;
			
		p1 = self.randomObject(objs, -1)
		p2 = self.randomObject(objs, p1)
		
		(partL, partG, winL, winG) = self.partition(eps, objs, p1, p2)
		self.quickJoinWin(winL, winG, eps, constSmallNumber)
		self.quickJoin(partL, eps, constSmallNumber)
		self.quickJoin(partG, eps, constSmallNumber)

	def quickJoinWin(self, objs1, objs2, eps, constSmallNumber):
		totalLen = len(objs1) + len(objs2)
		if(totalLen < constSmallNumber):
			self.nestedLoop2(eps, objs1, objs2)
			return;
		allObjects = objs1 + objs2
		p1 = self.randomObject(allObjects, -1)
		p2 = self.randomObject(allObjects, p1)

		(partL1, partG1, winL1, winG1) = self.partition(eps, objs1, p1, p2)
		(partL2, partG2, winL2, winG2) = self.partition(eps, objs2, p1, p2)

		self.quickJoinWin(winL1, winG2, eps, constSmallNumber)
		self.quickJoinWin(winG1, winL2, eps, constSmallNumber)
		self.quickJoinWin(partL1, partL2, eps, constSmallNumber)
		self.quickJoinWin(partG1, partG2, eps, constSmallNumber)

class quickDBSCANMongo(mongoConnect.MongoDBConnector):

	def __init__(self, dbname, host='localhost', port=27017):
		mongoConnect.MongoDBConnector.__init__(self, dbname, host='localhost', port=27017)

	def upsertPixelValue(self, collection, filter, epsNeigh):
		print(filter)
		pixelRecord = self.getRecord("quickDBSCAN", filter, ["_id", "object", "epsNeighs"])
		#print("pixelRecord-----------------------------------------")
		#print(pixelRecord)
		newNeighs = {'epsNeighs':list()}

		if(pixelRecord is not None):
			pixelNeighs = pixelRecord["epsNeighs"]
			alreadyExists = False
			for pixelNeigh in pixelNeighs:
				if(pixelNeigh==epsNeigh):
					alreadyExists = True
			if(alreadyExists==False):
				pixelNeighs.append(epsNeigh)
				newNeighs['epsNeighs'] = pixelNeighs;
		else:
			newNeighs['epsNeighs'].append(epsNeigh)

		self.db[collection].update(filter, {"$set":newNeighs}, upsert=True)




if __name__ == '__main__':

	filePath = sys.argv[1]
		
	quickDBSCANInstance = quickDBSCAN(filePath)
	quickDBSCANInstance.quickJoin(quickDBSCANInstance.pixelList, 10, 100)
	quickDBSCANInstance.mongoConnectInstance.closeConection()

