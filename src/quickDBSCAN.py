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

class QuickJOIN:
	
	def __init__(self, filePath):
		img = cv2.imread(filePath, cv2.IMREAD_COLOR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		imgReshaped = np.reshape(img, (len(img)*len(img[0]), 3))
		self.pixelList = list(np.unique(imgReshaped, axis=0))
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB");
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
					self.upsertPixelValue("quickDBSCAN",{"object":pixel1.tolist()}, pixel2.tolist())
					self.upsertPixelValue("quickDBSCAN",{"object":pixel2.tolist()}, pixel1.tolist())

	def nestedLoop2(self, eps, objs1, objs2):
		print("Nested loop 2")
		for pixel1 in objs1:
			for pixel2 in objs2:
				if( (pixel1 == pixel2).all() == False and self.euclideanDist(pixel1, pixel2) <= eps):
					print("Adauga")
					print(pixel1, pixel2)
					#insert into Mongo
					self.upsertPixelValue("quickDBSCAN",{"object":pixel1.tolist()}, pixel2.tolist())
					self.upsertPixelValue("quickDBSCAN",{"object":pixel2.tolist()}, pixel1.tolist())


	def randomObject(self, objs, pixel):
		if(type(pixel) is not int):
			#print(objs)
			objs = [x for x in objs if (x == pixel).all() == False]
		
		randomIndex = randint(0, len(objs)-1)

		return objs[randomIndex]

	def ball_average(self, objs, p1):
		avgDistHelper = []
		for pixel in objs:
			if( (pixel==p1).all() == False ):
				avgDistHelper.append(self.euclideanDist(pixel, p1))
		avgDistHelper = np.array(avgDistHelper)
		return sum(avgDistHelper)/len(avgDistHelper)

					
	def partition(self, eps, objs, p1, p2):
		partL = [] 
		partG = []
		winL = []
		winG = []
		r = self.ball_average(objs, p1)
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
		print("Intra in win")
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

	def upsertPixelValue(self, collection, filter, epsNeigh):
		#print(filter)
		pixelRecord = self.mongoConnectInstance.getRecord("quickDBSCAN", filter, ["_id", "object", "epsNeighs"])
		
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

		self.mongoConnectInstance.update("quickDBSCAN", filter, {"$set":newNeighs})
			

class DBSCAN:
	def __init__(self, eps, minPts):
		self.eps = eps
		self.minPts = minPts
		self.label = 0
		self.pixelLabels = collections.defaultdict(list);
		self.visitedPixels = []
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB");

	def checkIfPixelInList(self, pixel, objList):
		for pixelList in objList:
			if( (pixel==pixelList).all() == True ):
				return True
		return False


	def dbscan(self, objs):
		for pixel in objs:
			if(self.checkIfPixelInList(pixel, self.visitedPixels) == False):
				pixelRecord = self.mongoConnectInstance.getRecord("quickDBSCAN", {'object':pixel.tolist()}, ["_id", "object", "epsNeighs"])
				if(pixelRecord is not None):
					self.visitedPixels.append(pixel);
					spherePoints = pixelRecord["epsNeighs"]
					spherePoints = [np.asarray(tuple(p)) for p in spherePoints]
					#print(spherePoints)
					#print("len Sphere "+str(len(spherePoints)))
					if(len(spherePoints) >= self.minPts):
						self.label = self.label + 1
						print("Cluster "+str(self.label))
						self.pixelLabels[tuple(pixel)] = self.label
						self.expandCluster(spherePoints)
		return self.pixelLabels

	def expandCluster(self, spherePoints):
		print("Intra in expand")
		for pixel in spherePoints:
			#print(self.visitedPixels)
			#print("pixelul "+str(pixel)+" continut in "+str(self.checkIfPixelInList(pixel, self.visitedPixels)))
			if(self.checkIfPixelInList(pixel, self.visitedPixels) == False):
				print("Trece de check")
				self.visitedPixels.append(pixel)
				self.pixelLabels[tuple(pixel)] = self.label
				pixelRecord = self.mongoConnectInstance.getRecord("quickDBSCAN", {'object':pixel.tolist()}, ["_id", "object", "epsNeighs"])
				if(pixelRecord is not None):
					print("Intra")
					spherePointsNeighs = pixelRecord["epsNeighs"]
					spherePointsNeighs = [np.asarray(tuple(p)) for p in spherePointsNeighs]
					print("len Sphere vecini "+str(len(spherePointsNeighs)))
					if(len(spherePointsNeighs) >= self.minPts):
						self.expandCluster(spherePointsNeighs)
		return


if __name__ == '__main__':

	filePath = sys.argv[1]
		
	quickJoinInstance = QuickJOIN(filePath)
	'''quickJoinInstance.quickJoin(quickJoinInstance.pixelList, 10, 100)
	quickJoinInstance.mongoConnectInstance.closeConection()'''
	dbscanInstance = DBSCAN(10, 7)
	labelDict = dbscanInstance.dbscan(quickJoinInstance.pixelList)
	print(len(labelDict))
	print(set(labelDict.values()))

