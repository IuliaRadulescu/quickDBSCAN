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
		self.pixelList = []
		for i in range(len(img)):
			for j in range(len(img[0])):
				self.pixelList.append([i, j, img[i][j]]) #position x, position y, rgb code (numpy array)

		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB");

	
	def euclideanDistPosition(self, pixel1, pixel2):

		positionCoords1 = np.asarray([pixel1[0], pixel1[1]])
		positionCoords2 = np.asarray([pixel2[0], pixel2[1]])

		return np.linalg.norm(positionCoords1-positionCoords2)

		
	def nestedLoop(self, eps, objs):

		for pixel1 in objs:
			for pixel2 in objs:
				if(pixel1 != pixel2 and self.euclideanDistPosition(pixel1, pixel2) <= eps):
					#print(pixel1, pixel2)
					#insert into Mongo
					self.upsertPixelValue("quickDBSCAN",{"object":[pixel1[0], pixel1[1], pixel1[2].tolist()]}, [pixel2[0], pixel2[1], pixel2[2].tolist()])
					self.upsertPixelValue("quickDBSCAN",{"object":[pixel2[0], pixel2[1], pixel2[2].tolist()]}, [pixel1[0], pixel1[1], pixel1[2].tolist()])


	def nestedLoop2(self, eps, objs1, objs2):
		
		for pixel1 in objs1:
			for pixel2 in objs2:
				if( pixel1 != pixel2 and self.euclideanDistPosition(pixel1, pixel2) <= eps):
					print("Adauga")
					#print(pixel1, pixel2)
					#insert into Mongo
					self.upsertPixelValue("quickDBSCAN",{"object":[pixel1[0], pixel1[1], pixel1[2].tolist()]}, [pixel2[0], pixel2[1], pixel2[2].tolist()])
					self.upsertPixelValue("quickDBSCAN",{"object":[pixel2[0], pixel2[1], pixel2[2].tolist()]}, [pixel1[0], pixel1[1], pixel1[2].tolist()])


	def randomObject(self, objs):
		
		randomIndex = randint(0, len(objs)-1)

		return objs[randomIndex]

	def ball_average(self, eps, objs, p1):
		avgDistHelper = []
		for pixel in objs:
			if(pixel!=p1):
				avgDistHelper.append(self.euclideanDistPosition(pixel, p1))
		avgDistHelper = np.array(avgDistHelper)
		return sum(avgDistHelper)/len(avgDistHelper)

					
	def partition(self, eps, objs, p1):
		partL = [] 
		partG = []
		winL = []
		winG = []
		
		r = self.ball_average(eps, objs, p1)
		startIdx = 0
		endIdx = len(objs)-1
		startDist = self.euclideanDistPosition(objs[startIdx], p1)
		endDist = self.euclideanDistPosition(objs[endIdx], p1)
		
		while(startIdx < endIdx):
			while(endDist > r and startIdx < endIdx):
				if(endDist <= r+eps):
					winG.append(objs[endIdx])
				endIdx = endIdx - 1
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
				
			while(startDist <= r and startIdx < endIdx):
				if(startDist >= r-eps):
					winL.append(objs[startIdx])
				startIdx = startIdx + 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				
			if(startIdx < endIdx):
				if(endDist >= r-eps):
					winL.append(objs[endIdx])
				if(startDist <= r+eps):
					winG.append(objs[startIdx])
				#exchange items
				objs[startIdx], objs[endIdx] = objs[endIdx], objs[startIdx]
				startIdx = startIdx + 1
				endIdx = endIdx - 1
				startDist = self.euclideanDistPosition(objs[startIdx], p1)
				endDist = self.euclideanDistPosition(objs[endIdx], p1)
		
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
			
		p1 = self.randomObject(objs)
		
		(partL, partG, winL, winG) = self.partition(eps, objs, p1)
		if(len(winL)>0 and len(winG)>0):
			self.quickJoinWin(eps, winL, winG, constSmallNumber)
		if(len(partG)>0):
			self.quickJoin(eps, partL, constSmallNumber)
		if(len(partL)>0):
			self.quickJoin(eps, partG, constSmallNumber)

	def quickJoinWin(self, eps, objs1, objs2, constSmallNumber):
		print("Intra in win")
		totalLen = len(objs1) + len(objs2)
		if(totalLen < constSmallNumber):
			self.nestedLoop2(eps, objs1, objs2)
			return;
		allObjects = objs1 + objs2
		p1 = self.randomObject(allObjects)

		(partL1, partG1, winL1, winG1) = self.partition(eps, objs1, p1)
		(partL2, partG2, winL2, winG2) = self.partition(eps, objs2, p1)

		self.quickJoinWin(eps, winL1, winG2, constSmallNumber)
		self.quickJoinWin(eps, winG1, winL2, constSmallNumber)
		self.quickJoinWin(eps, partL1, partL2, constSmallNumber)
		self.quickJoinWin(eps, partG1, partG2, constSmallNumber)

	'''def upsertPixelValue(self, collection, filter, epsNeigh):
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

		self.mongoConnectInstance.update("quickDBSCAN", filter, {"$set":newNeighs})'''

	def upsertPixelValue(self, collection, filter, epsNeigh):

		self.mongoConnectInstance.update("quickDBSCAN", filter, {"$push":{"epsNeighs":epsNeigh}})
			

class DBSCAN:
	def __init__(self, eps, minPts, filePath):
		self.filePath = filePath
		self.minPts = minPts
		self.label = 0
		self.pixelLabels = collections.defaultdict(list);
		self.visitedPixels = []
		self.mongoConnectInstance = mongoConnect.MongoDBConnector("QuickDBScanDB");

	def checkIfPixelInList(self, pixel, objList):
		for pixelList in objList:
			if(pixel==pixelList):
				return True
		return False


	def dbscan(self, objs):
		for pixel in objs:
			pixel[2] = pixel[2].tolist()
			if(self.checkIfPixelInList(pixel, self.visitedPixels) == False):
				self.visitedPixels.append(pixel)
				pixelRecord = self.mongoConnectInstance.getRecord("quickDBSCAN", {'object':[pixel[0], pixel[1], pixel[2]]}, ["_id", "object", "epsNeighs"])
				if(pixelRecord is not None):
					spherePoints = pixelRecord["epsNeighs"]
					#print(spherePoints)
					#print("len Sphere "+str(len(spherePoints)))
					if(self.computeColorMinPts(spherePoints) >= self.minPts):
						self.label = self.label + 1
						print("Cluster "+str(self.label))
						self.pixelLabels[self.label].append(pixel)
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
				self.pixelLabels[self.label].append(pixel)
				pixelRecord = self.mongoConnectInstance.getRecord("quickDBSCAN", {'object':[pixel[0], pixel[1], pixel[2]]}, ["_id", "object", "epsNeighs"])
				if(pixelRecord is not None):
					print("Intra")
					spherePointsNeighs = pixelRecord["epsNeighs"]
					print("len Sphere vecini "+str(len(spherePointsNeighs)))
					minPtsComp = self.computeColorMinPts(spherePointsNeighs)
					if(minPtsComp >= self.minPts):
						print("Este mai mare expand "+str(minPtsComp))
						self.expandCluster(spherePointsNeighs)
		return

	def computeColorEps(self):
		img = cv2.imread(self.filePath, cv2.IMREAD_COLOR)

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

		return [bRadius, gRadius, rRadius]

	def distanceEps(self, eps, pixel1, pixel2):
		
		bterm = pow( (pixel1[0] - pixel2[0]), 2 )/pow( eps[0], 2 )
		gterm = pow( (pixel1[1] - pixel2[1]), 2 )/pow( eps[1], 2 )
		rterm = pow( (pixel1[2] - pixel2[2]), 2 )/pow( eps[2], 2 )

		return bterm + gterm + rterm

	def computeColorMinPts(self, objs):
		minPts = [];
		eps = self.computeColorEps()
		for pixelId1 in range(len(objs)):
			for pixelId2 in range(pixelId1+1, len(objs)):
				rgb1 = objs[pixelId1][2]
				rgb2 = objs[pixelId2][2]
				if(rgb1!=rgb2):
					if(self.distanceEps(eps, rgb1, rgb2) <= 1):
						minPts.append(tuple(rgb1))
						minPts.append(tuple(rgb2))
		return len(set(minPts))



def random_color():
	b = randint(0, 255)
	g = randint(0, 255)
	r = randint(0, 255)
	return [b, g, r]

def getLabelByKey(dict, searchValue):
	for theKey, theValue in dictionary.items():
		if(theValue == searchValue):
			return theKey
	return -1

def saveClusteredImage(labelDict, filePath, imageWidth, imageHeight):

	print("Save clustered Image")
	img = cv2.imread(filePath, cv2.IMREAD_COLOR)
	imageWidth = len(img[0])
	imageHeight = len(img)
	imagePixels = np.reshape(img, (imageWidth*imageHeight, 3))

	colorDict = collections.defaultdict(list);

	for pixelLabel in set(labelDict.keys()):
		colorDict[pixelLabel] = random_color();

	for pixelId in range(len(imagePixels)):
		pixelToChange = imagePixels[pixelId].tolist()
		pixelLabel = getLabelByKey(labelDict, pixelToChange)

		if(pixelLabel == -1):
			color = [0, 0, 0]
		else:
			color = colorDict[pixelLabel]
		imagePixels[pixelId][0] = color[0]
		imagePixels[pixelId][1] = color[1]
		imagePixels[pixelId][2] = color[2]
	final_image = np.reshape(imagePixels, (imageHeight, imageWidth, 3))

	cv2.imwrite('/home/iulia/CSCS/img/result.jpg', final_image)



if __name__ == '__main__':

	sys.setrecursionlimit(15000)

	filePath = sys.argv[1]
		
	quickJoinInstance = QuickJOIN(filePath)

	#quickJoinInstance.quickJoin(5, quickJoinInstance.pixelList, 500)
	#quickJoinInstance.mongoConnectInstance.closeConection()
	dbscanInstance = DBSCAN(5, 40, filePath)
	labelDict = dbscanInstance.dbscan(quickJoinInstance.pixelList)
	print(len(labelDict))
	print(set(labelDict.keys()))
	saveClusteredImage(labelDict, filePath, quickJoinInstance.imageWidth, quickJoinInstance.imageHeight)

