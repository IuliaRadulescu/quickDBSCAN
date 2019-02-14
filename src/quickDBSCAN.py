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
from scipy.signal import argrelextrema
import pymongo

import mongoConnect

class QuickJOIN:
	
	def __init__(self, filePath):
		img = cv2.imread(filePath, cv2.IMREAD_COLOR)
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		self.imageWidth = len(img[0])
		self.imageHeight = len(img)
		self.imgReshaped = np.reshape(img, (len(img)*len(img[0]), 3))

		self.pixelList = list(np.unique(self.imgReshaped, axis=0))
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB");
		#print(self.pixelList)

	
	def euclideanDist(self, pixel1, pixel2):
		#print(type(pixel1))
		#print(type(pixel2))
		return np.linalg.norm(pixel1-pixel2)

	def distanceEps(self, eps, pixel1, pixel2):
		
		bterm = pow( (pixel1[0] - pixel2[0]), 2 )/pow( eps[0], 2 )
		gterm = pow( (pixel1[1] - pixel2[1]), 2 )/pow( eps[1], 2 )
		rterm = pow( (pixel1[2] - pixel2[2]), 2 )/pow( eps[2], 2 )

		return bterm + gterm + rterm

		
	def nestedLoop(self, eps, objs):

		for pixel1 in objs:
			for pixel2 in objs:
				if( (pixel1 == pixel2).all() == False and self.euclideanDist(pixel1, pixel2) <= eps):
					print(pixel1, pixel2)
					#insert into Mongo
					self.upsertPixelValue("quickDBSCAN",{"object":pixel1.tolist()}, pixel2.tolist())
					self.upsertPixelValue("quickDBSCAN",{"object":pixel2.tolist()}, pixel1.tolist())


	def nestedLoop2(self, eps, objs1, objs2):
		
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

	def ball_average(self, eps, objs, p1):
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
		
		r = self.ball_average(eps, objs, p1)
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

	
				
		
	def quickJoin(self, eps, objs, constSmallNumber):
		if(len(objs) < constSmallNumber):
			self.nestedLoop(eps, objs)
			return;
			
		p1 = self.randomObject(objs, -1)
		p2 = self.randomObject(objs, p1)
		
		(partL, partG, winL, winG) = self.partition(eps, objs, p1, p2)
		self.quickJoinWin(eps, winL, winG, constSmallNumber)
		self.quickJoin(eps, partL, constSmallNumber)
		self.quickJoin(eps, partG, constSmallNumber)

	def quickJoinWin(self, eps, objs1, objs2, constSmallNumber):
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

		self.quickJoinWin(eps, winL1, winG2, constSmallNumber)
		self.quickJoinWin(eps, winG1, winL2, constSmallNumber)
		self.quickJoinWin(eps, partL1, partL2, constSmallNumber)
		self.quickJoinWin(eps, partG1, partG2, constSmallNumber)

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
	def __init__(self, minPts):
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
				self.visitedPixels.append(pixel)
				pixelRecord = self.mongoConnectInstance.getRecord("quickDBSCAN", {'object':pixel.tolist()}, ["_id", "object", "epsNeighs"])
				if(pixelRecord is not None):
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


def random_color():
	b = randint(0, 255)
	g = randint(0, 255)
	r = randint(0, 255)
	return [b, g, r]

def saveClusteredImage(labelDict, imagePixels, imageWidth, imageHeight):

	colorDict = collections.defaultdict(list);

	for pixelLabel in set(labelDict.values()):
		colorDict[pixelLabel] = random_color();

	for pixelId in range(len(imagePixels)):
		pixelToChange = tuple(imagePixels[pixelId])
		print(pixelToChange)
		if(labelDict[pixelToChange] == []):
			color = [0, 0, 0]
		else:
			color = colorDict[labelDict[pixelToChange]]
		imagePixels[pixelId][0] = color[0]
		imagePixels[pixelId][1] = color[1]
		imagePixels[pixelId][2] = color[2]
	final_image = np.reshape(imagePixels, (imageHeight, imageWidth, 3))

	cv2.imwrite('/home/iulia/CSCS/img/result.jpg', final_image)

def computeMinPts(imageWidth, imageHeight, percent):
	return int((imageWidth*imageHeight*percent)/255)


def computeEps(filePath):
	img = cv2.imread(filePath, cv2.IMREAD_COLOR)

	histB = cv2.calcHist([img],[0],None,[256],[0,256])
	localMinB = argrelextrema(histB, np.less)
	localDiffB = np.diff(localMinB[0])
	bRadius = sum(localDiffB)/len(localDiffB)

	histG = cv2.calcHist([img],[1],None,[256],[0,256])
	localMinG = argrelextrema(histG, np.less)
	localDiffG = np.diff(localMinG[0])
	gRadius = sum(localDiffG)/len(localDiffG)

	histR = cv2.calcHist([img],[1],None,[256],[0,256])
	localMinR = argrelextrema(histR, np.less)
	localDiffR = np.diff(localMinR[0])
	rRadius = sum(localDiffR)/len(localDiffR)

	scaled = np.linalg.norm(np.array([bRadius, gRadius, rRadius]) - np.zeros(3)) 

	return scaled




if __name__ == '__main__':

	filePath = sys.argv[1]
		
	quickJoinInstance = QuickJOIN(filePath)

	eps = computeEps(filePath)
	minPts = computeMinPts(quickJoinInstance.imageWidth, quickJoinInstance.imageHeight, 0.5)

	quickJoinInstance.quickJoin(eps, quickJoinInstance.pixelList, 200)
	quickJoinInstance.mongoConnectInstance.closeConection()
	dbscanInstance = DBSCAN(5)
	labelDict = dbscanInstance.dbscan(quickJoinInstance.pixelList)
	print(len(labelDict))
	print(set(labelDict.values()))
	saveClusteredImage(labelDict, quickJoinInstance.imgReshaped, quickJoinInstance.imageWidth, quickJoinInstance.imageHeight)

