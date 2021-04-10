#!/usr/bin/python3

import csv,copy
import numpy as np


class BINCode(object):

    def __init__(self,width = 10,dataType="binary"):
        self.CODE_COUNT = 15
        self.CODE_WIDTH = width
        self.LABELS_COUNT = 10
        self.ALL_CODES = pow(2,self.CODE_WIDTH)
        self.code = [-1]*self.CODE_COUNT
        self.labels = [-1]*self.CODE_COUNT
        self.dataType = dataType


    def GetDataType(self):
        return  self.dataType

    def CodeWidth(self):
        return self.CODE_WIDTH

    def AllCodesCount(self):
        return self.ALL_CODES
    def Labels(self):
        return self.labels


    def CreateTranslateTableAliases(self):

        aliases = {}
        for l,c in self.LabelCode().items():
            aliases[l] = [c]

        return aliases

    def TranslateTableAliasesCount(self):
        count = 0
        aliases = self.CreateTranslateTableAliases()
        for c in aliases.values():
            count += len(c)

        return count

    def CreateOneHotClasses(self):
        aliases = self.CreateTranslateTableAliases()

        sortedByLabel = sorted(aliases.items(), key=lambda x: x[0])

        oneHotCodesCount = self.TranslateTableAliasesCount()

        posIndex = 0
        oneHotCodes = []
        for label,codes in sortedByLabel:
            for c in codes:
                numPosIndex = 1 << posIndex
                binPosIndex = self.Num2BinArray(numPosIndex,oneHotCodesCount,np.float32)

                oneHotCodes.append( [label,c,posIndex,numPosIndex,binPosIndex] )
                posIndex += 1

        return oneHotCodes

    def PrecisionOneHotAll(self,predictions,labels):
        acc = self.PrecisionOneHot(predictions,labels)

        return acc,100.0 - acc,acc


    def PrecisionOneHot(self,predictions,labels):

        data = predictions
        m = np.argmax(data,axis = 1)
        data[:,:] = 0
        data[np.arange(len(predictions)),m] = 1



        tp = 0
        fp = 0
        for pred,numLabel in zip(data,labels):

            numPred  = self.BinArray2Num(pred)

            predLabel = self.LabelIDByCode(numPred)
            if predLabel == numLabel:
                tp += 1
            else:
                fp +=1

        acc = (tp / (tp+fp)) * 100.0
        return  acc

    def PrecisionBinary(self,predictions,labels):

        predictions[predictions <  0.5] = 0.0
        predictions[predictions >= 0.5] = 1.0
        predictions = predictions.astype('int8')
        tp = 0
        fp = 0
        for pred,numLabel in zip(predictions,labels):

            numPred  = self.BinArray2Num(pred)

            predLabel = self.LabelIDByCode(numPred)
            if predLabel == numLabel:
                tp += 1
            else:
                fp +=1

        acc = (tp / (tp+fp)) * 100.0
        return  acc

    def PrecisionBinaryAll(self,predictions,labels):

        num_all = len(predictions)

        labelAliases = self.CreateTranslateTableAliases()

        labelAliasesNegative = {}

        for olp in labelAliases.keys():
            labelAliasesNegative[olp] = []
            for oln,cs in labelAliases.items():
                if olp != oln:
                    labelAliasesNegative[olp].extend(cs)

        predictions[predictions <  0.5] = 0.0
        predictions[predictions >= 0.5] = 1.0
        binPredictions = predictions.astype('int8')

        count = 0
        ok = 0
        detected = 0
        undetected = 0
        for n,(pred,origLabel) in enumerate(zip(binPredictions,labels)):
            count += 1

            numPred  = self.BinArray2Num(pred)
            codes = labelAliases[origLabel]
            if numPred in codes:
                ok += 1

            elif numPred in labelAliasesNegative[origLabel]:
                undetected += 1
            else:
                detected += 1


        num_acceptable = ok + undetected

        realiab =  (ok / num_acceptable) * 100.0

        accuracy = ((ok)/num_all)* 100.0
        detectecd = ((detected)/num_all)* 100.0
        undetected_acc= ((undetected)/num_all)* 100.0

        return  accuracy,undetected_acc,realiab


    def PrecisionEuclideSoftmaxAll(self,data,labels):
        acc = self.PrecisionOneHotZadeh(data,labels)
        return  acc,100.0-acc,acc


    def PrecisionOneHotZadeh(self,data,labels):

            m = np.argmax(data,axis = 1)
            data[:,:] = 0
            data[np.arange(len(data)),m] = 1

            oneHotLabelAliases = self.CreateOneHotClasses()
            count = 0
            tp = 0
            for n,(oneHotPred,lab) in enumerate(zip(data,labels)):
                count += 1
                for items in oneHotLabelAliases:
                    if np.array_equal(oneHotPred,items[4]):
                        if lab == items[0]:
                            tp += 1
                            break

            return (tp/count)*100.0

    def PrecisionOneHotZadehAll(self,predictions,labels):
        data = predictions

        #hard coded!!!
        width = 1024
        outWidth = 10
        dataMax = (width - 1) - np.argmax(data,axis = 1)

        predictionsBin = np.zeros( (dataMax.shape[0] , outWidth), dtype=np.int8)

        labelAliases = self.CreateTranslateTableAliases()


        labelAliasesNegative = {}

        for olp in labelAliases.keys():
            labelAliasesNegative[olp] = []
            for oln,cs in labelAliases.items():
                if olp != oln:
                    labelAliasesNegative[olp].extend(cs)

        count = 0
        ok = 0
        repairable = 0
        unrepairable = 0
        for n,(pred,origLabel) in enumerate(zip(dataMax,labels)):
            count += 1

            codes = labelAliases[origLabel]
            if pred in codes:
                ok += 1

            elif pred in labelAliasesNegative[origLabel]:
                unrepairable += 1
            else:
                repairable += 1

            predictionsBin[n] = self.Num2BinArray(pred,outWidth,type=np.int8)

        fca = (ok/count)*100.0
        detectable = (repairable/count)*100.0
        undetectable_acc = (unrepairable/count)*100.0


        num_acceptable = ok + unrepairable
        realiability =  (ok / num_acceptable) * 100.0

        return fca,undetectable_acc,realiability


    def PrecisionHadamardEuclideAll(self,data,labels):

        predictions = data
        num_all = len(predictions)

        labelAliases = self.CreateTranslateTableAliases()

        labelAliasesNegative = {}

        for olp in labelAliases.keys():
            labelAliasesNegative[olp] = []
            for oln,cs in labelAliases.items():
                if olp != oln:
                    labelAliasesNegative[olp].extend(cs)


        allCodesCount = self.AllCodesCount()
        allCodes = np.zeros((allCodesCount,self.CodeWidth()),dtype=np.float32)
        for ic in range(allCodesCount):
            z = self.Num2BinArrayF(ic)
            allCodes[ic] = z

        count = 0
        ok = 0
        repairable = 0
        unrepairable = 0

        for n,(pred,origLab) in enumerate(zip(predictions,labels)):

            all_euc = np.linalg.norm((pred-allCodes),axis=1)
            min_euc = np.argmin(all_euc)

            eucLabel = -1
            count += 1
            fp = False

            for codeLab,codeNums in labelAliases.items():
                if min_euc in codeNums:
                    eucLabel = codeLab
                    break

            if origLab == eucLabel:
                ok += 1
            else:

                for codeLab,codeNums in labelAliasesNegative.items():
                    if min_euc in codeNums:
                        fp = True
                        break

                if fp == True:
                    unrepairable += 1
                else:
                    repairable += 1


        num_acceptable = ok + unrepairable

        accuracy = ((ok)/num_all)* 100.0
        prec_repairable = ((repairable)/num_all)* 100.0
        prec_unrepairable = ((unrepairable)/num_all)* 100.0
        realiab =  (ok / num_acceptable) * 100.0

        return  accuracy,prec_unrepairable,realiab


    def PrecisionHadamardEuclide(self,predictions,labels):

        labelAliases = self.CreateTranslateTableAliases()
        count = 0
        ok = 0
        for n,(pred,origLab) in enumerate(zip(predictions,labels)):

            eucMin = 100000
            eucLabel = -1
            count += 1
            for codeLab,codeNums in labelAliases.items():
                for c in codeNums:
                    binCcodeNum = self.Num2BinArrayI(c)
                    euc = np.linalg.norm((pred-binCcodeNum))

                    if euc < eucMin:
                        eucMin = euc
                        eucLabel = codeLab

            if origLab == eucLabel:
                ok += 1

        return (ok/count)*100.0


    def ComputePrecision(self,predictions,labels):

        data = copy.deepcopy(predictions)
        if self.dataType == "one_hot":
            accuracy = self.PrecisionOneHot(data,labels)
        if self.dataType == "one_hot_all":
            accuracy = self.PrecisionOneHotAll(data,labels)
        elif self.dataType == "one_hot_zadeh_all":
            accuracy = self.PrecisionOneHotZadehAll(data,labels)
        elif self.dataType == "one_hot_zadeh" or self.dataType == "euclide_softmax":
            accuracy = self.PrecisionOneHotZadeh(data,labels)
        elif self.dataType == "euclide_softmax_all":
            accuracy = self.PrecisionEuclideSoftmaxAll(data,labels)
        elif self.dataType == "binary":
            accuracy = self.PrecisionBinary(data,labels)
        elif self.dataType == "binary_all":
            accuracy = self.PrecisionBinaryAll(data,labels)
        if self.dataType == "hadamard_euclide":
            accuracy = self.PrecisionHadamardEuclide(data,labels)
        if self.dataType == "hadamard_euclide_all":
            accuracy = self.PrecisionHadamardEuclideAll(data,labels)

        return  accuracy





    def CreateHMatrix(self,baseCode):

        H = copy.deepcopy(baseCode)

        H[0] = H[0] & 0b0000111111
        H[1] = H[1] & 0b0000111111
        H[2] = H[2] & 0b0000111111
        H[3] = H[3] & 0b0000111111

        for i in reversed(range(0,self.CODE_WIDTH-4)):
            x = 1 << i
            x = (x&0b0000111111)
            H.append( x)

        binH = []
        arrayH = np.zeros((6,10),dtype=np.uint8)
        for i,h in enumerate(H):
            p = self.Code2Bin(h,6)
            binH.append(p)
            for j,b in enumerate(p) :
                arrayH[j][i] = b

        return  arrayH

    @staticmethod
    def BinArray2Num(binNum):
        n = binNum.reshape(-1,)
        n = np.require(n,dtype=np.int8).tolist()
        n = map(str, n)
        n = "".join(n)
        n = int(n,2)
        return n

    @staticmethod
    def BinnarizeArray2Num(binNum):
        n = binNum.reshape(-1).tolist()
        n = map(str, n)
        n = "".join(n)
        n = int(n,2)
        return n

    def BinarizeDataSelf(self,data):
        return  BINCode.BinarizeData(data,dataType=self.dataType)

    @staticmethod
    def BinarizeData(data,dataType="binary"):
        if dataType == "one_hot":
            m = np.argmax(data,axis = 1)
            data[:,:] = 0
            data[np.arange(len(data)),m] = 1

        elif dataType == "euclide":
            pass
        elif dataType == "one_hot_all":

            maximumIndexes = np.argmax(data,axis = 1)
            outWidth = int(np.log2(data.shape[1]))
            output = np.zeros( (data.shape[0] , outWidth), dtype=np.int8)
            for n,max in enumerate(maximumIndexes):
                output[n] = BINCode.Num2BinArray(max,outWidth,type=np.int8)

            data = output

        elif dataType == "one_hot_select":

            maximumIndexes = 9-np.argmax(data,axis = 1)
            #print(maximumIndexes[:10])
            outWidth = data.shape[1]
            output = np.zeros( (data.shape[0] , outWidth), dtype=np.int8)
            for n,max in enumerate(maximumIndexes):

                #num = 1 << int( max )
                #labelAliases[max]
                output[n] = -1#BINCode.Num2BinArray(max,outWidth,type=np.int8)
                print("ERROR: BINARIZE DATA ERROR!!!!")

            data = output

        elif dataType == "binary":
            data[data <  0.5] = 0.0
            data[data >= 0.5] = 1.0
            data = data.astype('int8')
        else:
            print("ERROR: UNKNOWN data type: %s !!"%dataType )
            exit(1)


        #print(data[:2,:])
        return data

    @staticmethod
    def Num2BinArray(num,width,type=np.int8):

        arrayH = np.zeros((1,width),dtype=type)
        p = BINCode.Code2Bin(num,width)
        for i,b in enumerate(p) :
            arrayH[0][i] = b

        return arrayH.reshape((width))
        #return arrayH.reshape((width,1))

    def Num2BinArrayI(self,num):
        return BINCode.Num2BinArray(num,self.CODE_WIDTH,type=np.int8)
    def Num2BinArrayF(self,num):
        return BINCode.Num2BinArray(num,self.CODE_WIDTH,type=np.float32)

    def NonLabelsCode(self):
        noLabels = []

        for l,c in zip(self.labels,self.code):
            if l == -1:
                noLabels.append(c)
        noLabels.append(0)
        return  noLabels

    def NonLabelsAllCode(self):

        noLabels = []
        code = self.LabelCode()
        for i in range(self.ALL_CODES):
            if i not in code.values():
                noLabels.append(i)

        return  noLabels

    def AllCodes(self):

        noLabels = []
        for i in range(self.ALL_CODES):
            noLabels.append(i)
        return  noLabels

    def Code(self):
        return  self.code

    def LabelIDByCode(self,w):

        codes = self.LabelCode()
        for l,c in codes.items():
            if c == w:
                return l
        return  -1

    def ConvertLables2CodeWords(self,data):
        lc = self.LabelCode()
        for i,tl2 in enumerate(data):
            dataL = tl2
            if dataL in lc.keys() :
                data[i] = lc[dataL]

        return data

    def ConvertArrayNum2BinCode(self,data):

        code = self.LabelCode()

        binCode = np.zeros(( data.shape[0], self.CODE_WIDTH ),dtype=np.float32)
        for i,l in enumerate(data):
            binCode[i] = self.Num2BinArray(code[int(l)],self.CODE_WIDTH,np.float32)

        return  binCode

    def LabelCode(self):

        r = {}

        for i,n in enumerate(self.labels):
            if n < 0:
                continue
            r[n] = self.code[i]
        return r

    def SetCode(self,baseCode):
        self.code = self.CompouteLC(baseCode)

    def SetLabels(self,labels):
        self.labels = labels

    @staticmethod
    def Code2Bin(code,width=10):

        binCode = ("{0:0%db}"%width).format(code)
        return binCode


    @staticmethod
    def HammingDistance(code1,code2):

        a = code1 ^ code2
        b = bin(a)
        dist = b.count('1')
        return  dist

    def MSEDistance(self,code1,code2):


        c1 = BINCode.Num2BinArray(code1,self.CODE_WIDTH,np.float32)
        c2 = BINCode.Num2BinArray(code2,self.CODE_WIDTH,np.float32)

        a = np.square(c1 - c2)
        mse = a.mean()
        return  mse

    def EuclideDistance(self,code1,code2):

        c1 = BINCode.Num2BinArray(code1,self.CODE_WIDTH,np.float32)
        c2 = BINCode.Num2BinArray(code2,self.CODE_WIDTH,np.float32)

        d = np.linalg.norm((c1-c2))
        return d



    @staticmethod
    def ComputeCodeDistance(code,metric,triangle = False):

        codeList = code
        if isinstance(code,dict):
            so = sorted(code.items(),key=lambda x:x[0])
            codeList = [i[1] for i in so]

        dist = np.zeros((len(codeList),len(codeList)),dtype=np.float32)
        for i in range(len(codeList)):
            for j in range(len(codeList)):
                x = metric(codeList[i] ,codeList[j])
                if triangle and i > j:
                    x = 0
                dist[i][j] = x
        return  dist


    @staticmethod
    def ComputeCodeHammingDistance(code,triangle = False):

        codeList = code
        if isinstance(code,dict):
            so = sorted(code.items(),key=lambda x:x[0])
            codeList = [i[1] for i in so]

        dist = np.zeros((len(codeList),len(codeList)),dtype=np.int)
        for i in range(len(codeList)):
            for j in range(len(codeList)):
                a = codeList[i] ^ codeList[j]
                b = bin(a)
                x = b.count('1')
                if triangle and i > j:
                    x = 0


                dist[i][j] = x
        return  dist


    def SaveCode(self,file):

        f = open(file,"w")

        f.write("n;code;label;code_bin;\n")
        for i,c in enumerate(self.code):

            f.write("%2d;%4s;%3s; "%(i,c,self.labels[i]))

            p = self.Code2Bin(c,self.CODE_WIDTH)
            for k in p:
                f.write("%s;"%(k))

            f.write("\n")
        f.close()


    def LabelsCount(self):
        return self.LABELS_COUNT

    def LoadCode(self,path):

        csvData = {}

        with open(path,'r') as fr:
            reader =  csv.reader(fr,delimiter=';')
            for i,row in enumerate(reader):
                try:
                    int(row[0])
                except:
                    continue
                num = int(row[0])
                code = int(row[1])
                label = int(row[2])
                csvData[num] = [code,label]

            self.code.clear()
            self.labels.clear()
            for i in csvData.keys():
                self.code.append(csvData[i][0])
                self.labels.append(csvData[i][1])

            return True


        print("ERROR: Can't open code file %s"%path)
        return False


