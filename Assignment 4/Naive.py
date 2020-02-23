import collections
import pprint
import math

"""file_input = input('File Name: ')        trial.txt"""

#ENGLISH
filenameslistEnglish = ['e0.txt', 'e1.txt', 'e2.txt','e3.txt', 'e4.txt', 'e5.txt','e6.txt', 'e7.txt', 'e8.txt','e9.txt']
"""filenameslist = ['Trial.txt', 'Trial1.txt']"""

alphalist = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
DictEnglish = {'a':0,'b':0,'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,'m':0,'n':0,'o':0,'p':0,'q':0,'r':0,
        's':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0,' ':0}

totalEnglishchar = 0

for x in filenameslistEnglish:
    with open('C:/Users/akash/OneDrive/Documents/TrainingDataLanguage/'+x, 'r') as info:
        count = collections.Counter(info.read())
        
        for i in alphalist:
            DictEnglish[i] = DictEnglish[i] + count[i]
            
for x in DictEnglish.values(): 
    totalEnglishchar = totalEnglishchar + x             

#print(DictEnglish)
#print(totalEnglishchar)

charprobabilityEnglish = []

for x in DictEnglish.values():
    a = (x+1)/(totalEnglishchar+27)
    charprobabilityEnglish.append(a)

#print(charprobabilityEnglish)

#################################################################################################
#JAPANESE
filenameslistJapanese = ['j0.txt', 'j1.txt', 'j2.txt','j3.txt', 'j4.txt', 'j5.txt','j6.txt', 'j7.txt', 'j8.txt','j9.txt']
"""filenameslistJapanese = ['j0.txt', 'j1.txt']"""

DictJapanese = {'a':0,'b':0,'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,'m':0,'n':0,'o':0,'p':0,'q':0,'r':0,
        's':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0,' ':0}

for x in filenameslistJapanese:
    with open('C:/Users/akash/OneDrive/Documents/TrainingDataLanguage/'+x, 'r') as info:
        count = collections.Counter(info.read())
        
        for i in alphalist:
            DictJapanese[i] = DictJapanese[i] + count[i]
        
totalJapanesechar = 0
#print(DictJapanese)

for x in DictJapanese.values():
    totalJapanesechar = totalJapanesechar + x
    
#print(totalJapanesechar)

charprobabilityJapanese = []
for x in DictJapanese.values():
    charprobabilityJapanese.append((x+1)/(totalJapanesechar + 27))

#print(charprobabilityJapanese)
#################################################################################################        

#SPANISH
filenameslistSpanish = ['s0.txt', 's1.txt', 's2.txt','s3.txt', 's4.txt', 's5.txt','s6.txt', 's7.txt', 's8.txt','s9.txt']
"""filenameslistSpanish = ['s0.txt', 's1.txt']"""

DictSpanish = {'a':0,'b':0,'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,'m':0,'n':0,'o':0,'p':0,'q':0,'r':0,
        's':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0,' ':0}

for x in filenameslistSpanish:
    with open('C:/Users/akash/OneDrive/Documents/TrainingDataLanguage/'+x, 'r') as info:
        count = collections.Counter(info.read())
        
        for i in alphalist:
            DictSpanish[i] = DictSpanish[i] + count[i]
        
#print(DictSpanish)
totalSpanishchar = 0

for x in DictSpanish.values():
    totalSpanishchar = totalSpanishchar + x
    
#print(totalSpanishchar)

charprobabilitySpanish = []
for x in DictSpanish.values():
    charprobabilitySpanish.append((x+1)/(totalSpanishchar + 27))

#print(charprobabilitySpanish)

#################################################################################################

Dicte10 = {'a':0,'b':0,'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,'m':0,'n':0,'o':0,'p':0,'q':0,'r':0,
        's':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0,' ':0}

with open('C:/Users/akash/OneDrive/Documents/languageID/e10.txt', 'r') as info:
        count = collections.Counter(info.read())
        
        for i in alphalist:
            Dicte10[i] = Dicte10[i] + count[i]
        
#print(Dicte10)

#################################################################################################
#5 QUESTION

#print(charprobabilityEnglish)
newsum = 0

"""for x in charprobabilityEnglish:
    a = math.pow(x,DictEnglish[i].value)
    newsum = newsum + math.log10[a]"""
   
for i in range(len(charprobabilityEnglish)):
    #print(Dicte10[])
    a = math.log(charprobabilityEnglish[i],10)
    b = Dicte10[alphalist[i]]
    newsum = newsum + a*b

#print(newsum)
#print(pow(10,newsum))
#################################################################################################

#5Q2
#print(charprobabilitySpanish)
newsum = 0

"""for x in charprobabilityEnglish:
    a = math.pow(x,DictEnglish[i].value)
    newsum = newsum + math.log10[a]"""
   
for i in range(len(charprobabilitySpanish)):
    #print(Dicte10[])
    a = math.log(charprobabilitySpanish[i],10)
    b = Dicte10[alphalist[i]]
    newsum = newsum + a*b

#print(newsum)

#################################################################################################

#5Q3
#print(charprobabilityJapanese)
newsum = 0

"""for x in charprobabilityEnglish:
    a = math.pow(x,DictEnglish[i].value)
    newsum = newsum + math.log10[a]"""
   
for i in range(len(charprobabilityJapanese)):
    #print(Dicte10[])
    a = math.log(charprobabilityJapanese[i],10)
    b = Dicte10[alphalist[i]]
    newsum = newsum + a*b

print(newsum)


#################################################################################################
#Q6

#Epow(-3405)/3
#English is the predicted class,a s it has the highest probability