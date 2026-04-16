from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import numpy as np
import auxLay
import auxTr
import manaData

class AeRc_CvDet(tf.keras.Model):
    def __init__(self, si, chi, cho, ctt=2, rep=1, chf=1, l2kr=0, l1a=0, l2kd=0):
        super(AeRc_CvDet, self).__init__()
        
        self.initGloUnf = tf.keras.initializers.GlorotUniform()

        REG_KNL_RC = tf.keras.regularizers.l2(l2kr)
        REG_ACT = tf.keras.regularizers.l1(l1a)
        REG_KNL_DET = tf.keras.regularizers.l2(l2kd)
        
        self.ctt = ctt
        self.arrDpy = []
        self.dictLay = {}

        arrLayInfo = ['ctt', 'rep', 'last', 'greedy', 'save', 'bott']
        ObjDict2LayInfo = manaData.Dict2()

        nameLay = 'in'
        self.dictLay[nameLay] = tf.keras.layers.InputLayer(input_shape=(si, si, chi), name=nameLay)
        tupLayInfo = (-1, -1, False, True, False, False)
        ObjDict2LayInfo.crt(nameLay, arrLayInfo)
        ObjDict2LayInfo.ins(nameLay, tupLayInfo)
        self.arrDpy.append(nameLay)
     
        for i in range(ctt):
            cha = int(chi * chf * (2**i))

            for j in range(rep):
                if(j == 0):
                    strd = 2
                else:
                    strd = 1

                nameLay = 'pade{}-{}'.format(i, j)
                self.dictLay[nameLay] = auxLay.Pad2D(padding_number=1, mode='REFLECT')
                tupLayInfo = (i, j, False, True, False, False)
                ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                self.arrDpy.append(nameLay)

                nameLay = 'cve{}-{}'.format(i, j)
                self.dictLay[nameLay] = tf.keras.layers.Conv2D(filters=cha, kernel_size=3, strides=strd, activation='relu', kernel_regularizer=REG_KNL_RC, activity_regularizer=REG_ACT, name=nameLay)
                tupLayInfo = (i, j, True, True, True, False)
                ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                self.arrDpy.append(nameLay)

        for i in range(ctt-1, -1, -1):
            if(i == 0):
                cha = chi
                strd = 2
                
                nameLay = 'cvd{}-0'.format(i)
                self.dictLay[nameLay] = tf.keras.layers.Conv2DTranspose(filters=cha, kernel_size=3, strides=strd, activation='linear', kernel_regularizer=REG_KNL_RC, name=nameLay)
                tupLayInfo = (i, 0, True, True, True, False)
                ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                self.arrDpy.append(nameLay)

                nameLay = 'crod{}-{}'.format(i, j)
                self.dictLay[nameLay] = auxLay.Cro2D(cropping_number=1)
                tupLayInfo = (i, j, True, True, False, False)
                ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                self.arrDpy.append(nameLay)
            else:
                cha = int(chi * chf * (2**(i-1)))

                for j in range(rep):
                    if(j == 0):
                        if(i == (ctt-1)):
                            swtBott = True
                        else:
                            swtBott = False

                        strd = 2

                        nameLay = 'cvd{}-{}'.format(i, j)
                        self.dictLay[nameLay] = tf.keras.layers.Conv2DTranspose(filters=cha, kernel_size=3, strides=strd, activation='relu', kernel_regularizer=REG_KNL_RC, name=nameLay)
                        tupLayInfo = (i, j, True, True, True, swtBott)
                        ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                        ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                        self.arrDpy.append(nameLay)

                        nameLay = 'crod{}-{}'.format(i, j)
                        self.dictLay[nameLay] = auxLay.Cro2D(cropping_number=1)
                        tupLayInfo = (i, j, True, True, False, False)
                        ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                        ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                        self.arrDpy.append(nameLay)
                    else:
                        strd = 1

                        nameLay = 'padd{}-{}'.format(i, j)
                        self.dictLay[nameLay] = auxLay.Pad2D(padding_number=1, mode='REFLECT')
                        tupLayInfo = (i, j, False, True, False, False)
                        ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                        ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                        self.arrDpy.append(nameLay)

                        nameLay = 'cvd{}-{}'.format(i, j)
                        self.dictLay[nameLay] = tf.keras.layers.Conv2D(filters=cha, kernel_size=3, strides=strd, activation='relu', kernel_regularizer=REG_KNL_RC, name=nameLay)
                        tupLayInfo = (i, j, True, True, True, False)
                        ObjDict2LayInfo.crt(nameLay, arrLayInfo)
                        ObjDict2LayInfo.ins(nameLay, tupLayInfo)
                        self.arrDpy.append(nameLay)

        nameLay = 'pad{}-{}'.format(i, j)
        self.dictLay[nameLay] = auxLay.Pad2D(padding_number=1, mode='REFLECT')
        tupLayInfo = (i, j, False, False, False, False)
        ObjDict2LayInfo.crt(nameLay, arrLayInfo)
        ObjDict2LayInfo.ins(nameLay, tupLayInfo)
        self.arrDpy.append(nameLay)

        nameLay = 'out'
        self.dictLay[nameLay] = tf.keras.layers.Conv2D(filters=cho, kernel_size=3, strides=1, activation='softmax', kernel_regularizer=REG_KNL_DET, name=nameLay)
        tupLayInfo = (-1, -1, False, False, True, False)
        ObjDict2LayInfo.crt(nameLay, arrLayInfo)
        ObjDict2LayInfo.ins(nameLay, tupLayInfo)
        self.arrDpy.append(nameLay)

        self.dict2LayInfo = ObjDict2LayInfo.get()
        self.arrGre = []

        for i in range(len(self.arrDpy)):
            if(self.dict2LayInfo[self.arrDpy[i]]['greedy']):
                self.arrGre.append(self.arrDpy[i])

        if(not(self.dict2LayInfo[self.arrGre[-1]]['last'])):
            self.arrGre = self.arrGre[:-1]

        self.ObjTr = auxTr.Train()

    def call(self, tsr):
        tsrIn = tf.identity(tsr)
        swtDet = 0

        for i in range(len(self.arrDpy)):
            if(self.dict2LayInfo[self.arrDpy[i]]['bott']):
                tsrLat = tf.identity(tsr)
            if(not(self.dict2LayInfo[self.arrDpy[i]]['greedy']) and (swtDet==0)):
                tsrRc = tf.identity(tsr)
                tsr = tf.math.subtract(tsr, tsrIn)
                swtDet += 1

            tsr = self.dictLay[self.arrDpy[i]](tsr)

        return tsrLat, tsrRc, tsr
    
    def reInit(self, swtRc, swtDet):
        if(swtRc == 1):
            for i in self.dict2LayInfo:
                if(self.dict2LayInfo[i]['greedy']):
                    for j in range(len(self.dictLay[i].trainable_weights)):
                        self.dictLay[i].trainable_weights[j].assign(self.initGloUnf(self.dictLay[i].trainable_weights[j].numpy().shape))

        if(swtDet == 1):
            for i in self.dict2LayInfo:
                if(not(self.dict2LayInfo[i]['greedy'])):
                    for j in range(len(self.dictLay[i].trainable_weights)):
                        self.dictLay[i].trainable_weights[j].assign(self.initGloUnf(self.dictLay[i].trainable_weights[j].numpy().shape))

    def confTr(self, swtRc, swtDet):
        if(swtRc == 0):
            statRc = False
        else:
            statRc = True

        if(swtDet == 0):
            statDet = False
        else:
            statDet = True
 
        for i in self.dict2LayInfo:
            if(self.dict2LayInfo[i]['greedy']):
                self.dictLay[i].trainable = statRc
            else:
                self.dictLay[i].trainable = statDet

    def confTrGre(self, cttTr):
        self.ObjTr.confGreedy(self.dictLay, self.dict2LayInfo, cttTr)

    def trGre(self, tsrIn, cttTr):

        return self.ObjTr.greedy(self.dictLay, self.dict2LayInfo, self.arrGre, cttTr, tsrIn)

    def saveWei(self, path):
        for i in self.dict2LayInfo:
            self.dictLay[i].trainable = True

            if(self.dict2LayInfo[i]['save']):
                if(i.count('bn')):
                    np.save('{}{}Beta.npy'.format(path, i), self.dictLay[i].trainable_weights[0].numpy())
                    np.save('{}{}Gamma.npy'.format(path, i), self.dictLay[i].trainable_weights[1].numpy())
                    np.save('{}{}Mean.npy'.format(path, i), self.dictLay[i].non_trainable_weights[0].numpy())
                    np.save('{}{}Var.npy'.format(path, i), self.dictLay[i].non_trainable_weights[1].numpy())
                else:
                    np.save('{}{}Wei.npy'.format(path, i), self.dictLay[i].trainable_weights[0].numpy())
                    np.save('{}{}Bias.npy'.format(path, i), self.dictLay[i].trainable_weights[1].numpy())

    def loadWei(self, path):
        for i in self.dict2LayInfo:
            if(self.dict2LayInfo[i]['save']):
                if(i.count('bn')):
                    vrb1 = np.load('{}{}Beta.npy'.format(path, i))
                    vrb2 = np.load('{}{}Gamma.npy'.format(path, i))
                    vrb3 = np.load('{}{}Mean.npy'.format(path, i))
                    vrb4 = np.load('{}{}Var.npy'.format(path, i))
                    arrWei = [vrb1, vrb2, vrb3, vrb4]
                else:
                    vrb1 = np.load('{}{}Wei.npy'.format(path, i))
                    vrb2 = np.load('{}{}Bias.npy'.format(path, i))
                    arrWei = [vrb1, vrb2]

                self.dictLay[i].set_weights(arrWei)
