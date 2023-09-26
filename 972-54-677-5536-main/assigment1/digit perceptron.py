from digit_trainer import digit

d=digit(196,2)
print("Start reading data")
#read data from train file
d.createbatch('digits_train.txt') 
print("End reading data and Start training")

#train NN
d.train(d.batchI,d.batchT)
print("End training and start reading testing data")

#read data from test file
d.createbatch('digits_test.txt')
print("End reading testing data and start testing")

#start testing NN
count=0
failc=0
for i in range(500,750):
    test=d.test(d.batchI[i])
    if test[0] and not test[1]:
        count=count+1
    elif (test[0]==test[1]):
        failc=failc+1

#print(f"number of success recognizing digit '2' out of 250 paterns is: {count} - {count/250*100}%")
print("%d misses / 250 = %.2f%%"%(250-count,(250-count)/250*100))
#print("counts test which yield similar 2 output (i.e 0,0 or 1,1): %d"%(failc))

countNo2=0
failc=0
for i in range(0,500):
    test=d.test(d.batchI[i])
    if not test[0] and test[1]:
        countNo2=countNo2+1
    elif (test[0]==test[1]):
        failc=failc+1
for i in range(750,2500):
    test=d.test(d.batchI[i])
    if not test[0] and test[1]:
        countNo2=countNo2+1
    elif(test[0]==test[1]):
        failc=failc+1

#print("number of success recognizing NOT digit '2' out of 2250 paterns is: %d - %.2f"%(countNo2, countNo2/2250*100))
print("%d flase positive / 2250 = %.2f%%"%(2250-countNo2,(2250-countNo2)/2250*100))
#print("counts test which yield similar 2 output (i.e 0,0 or 1,1): %d"%(failc))
