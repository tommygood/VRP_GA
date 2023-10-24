# make test data
import random

class writeData :

    def __init__(self, data) :
        self.data = data

    # 寫入 txt
    def txt(self, file_name) :
        f = open(file_name, "w")
        # total record num
        data_len = self.data[0]
        f.write(f"{data_len}\n")
        # each record
        for record in self.data[1:] :
            index = record[0]
            lat = record[1]
            lon = record[2]
            f.write(f"{index} {lat} {lon}\n")
        f.close()

def main() :
    data = makeTest(100)
    write = writeData(data)
    write.txt("delivery.txt")

def makeTest(num) :
    # 指標
    lat = 23
    lon = 120

    # test data
    data = [num]

    for i in range(num) :
        record = [i+1, lat + (random.random()*4 - random.random()*3)*0.02, lon + (random.random()*4 - random.random()*3)*0.02]
        data.append(record)
    return data

main()